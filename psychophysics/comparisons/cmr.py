import os
from glob import glob
import soundfile as sf
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import psychophysics.comparisons.cutil as cutil

conditions = ["mult", "rand"]


def generate_comparison_sounds(sound_group, settings, seeds, network, resample_rate=None):
    """ Comparison sounds to convert network soundwaves into judgments
        Instead of generating them, we can reconstruct them from the
        experiment stimuli.
    """

    sound_dir = os.path.join(os.environ["sound_dir"], sound_group, "")
    comparison_dict = {}
    for condition in conditions:
        comparison_dict[condition] = {}
        for bw in settings["bandwidths"]:
            comparison_dict[condition][bw] = {}
            for tone_level in settings["tone_levels"]:
                comparison_dict[condition][bw][tone_level] = {}
                for seed in seeds:
                    # Get premixture sounds
                    no_tone_name = f"{condition}_noTone_bw{bw}_seed{seed}-0"
                    mix_name = f"{condition}_tone_bw{bw}_toneLevel{tone_level}_seed{seed}-0"
                    only_noise, audio_sr = sf.read(sound_dir + no_tone_name + ".wav")
                    mix, audio_sr = sf.read(sound_dir + mix_name + ".wav")
                    only_tone = mix - only_noise
                    comparison_dict[condition][bw][tone_level][seed] = []
                    for sound in [only_noise, only_tone, mix, np.zeros(mix.shape)]:
                        if "sequential-gen-model" in network:
                            # Account for pad from amortized neural network duration requirement
                            seconds_per_frame = 0.005
                            if len(sound) % int(np.round(seconds_per_frame*audio_sr)) != 0:
                                N = int(np.round(seconds_per_frame*audio_sr))
                                diff = int(np.ceil(len(sound) / N) * N) - len(sound)
                                sound = np.concatenate((sound, np.zeros((diff,))))
                        if resample_rate is not None:
                            wav_rs = scipy.signal.resample_poly(sound, resample_rate, audio_sr)
                            gram = cutil.make_gammatonegram(wav_rs, resample_rate)
                        else:
                            gram = cutil.make_gammatonegram(sound, audio_sr)
                        comparison_dict[condition][bw][tone_level][seed].append(gram)
                         
    return comparison_dict


def get_distances(net_results, settings, comparison_dict, seeds):
    """ Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")
    judgments = {}
    for condition in conditions:
        judgments[condition] = {}
        for bw in settings["bandwidths"]:
            judgments[condition][bw] = {}
            for tone_level in settings["tone_levels"]:
                judgments[condition][bw][tone_level] = []
                for seed in seeds:
                    network_outputs = net_results[condition][bw][tone_level][seed]
                    comparison_sounds = comparison_dict[condition][bw][tone_level][seed] # Noise, tone, mix, silence
                    mix_gram = comparison_dict[condition][bw][tone_level][seed][2]
                    silence_gram = comparison_dict[condition][bw][tone_level][seed][3]

                    # Get distance for normalization in denominator
                    # Distance of standards to each other
                    normalizing_distances = np.full((2,2), np.nan)
                    for n_idx_1 in range(2):
                        norm_gram_1 = mix_gram if n_idx_1 == 0  else silence_gram
                        for n_idx_2 in range(2):
                            norm_gram_2 = mix_gram if n_idx_2 == 0  else silence_gram
                            if n_idx_1 == n_idx_2:
                                normalizing_distances[n_idx_1, n_idx_2] = np.inf
                            else:
                                # Compare norm_gram_1 to noise
                                d1 = distance_metric(comparison_sounds[0], norm_gram_1, comparison_sounds[0]>20)
                                # Compare norm_gram_2 to tone
                                d2 = distance_metric(comparison_sounds[1], norm_gram_2, comparison_sounds[1]>20)
                                normalizing_distances[n_idx_1, n_idx_2] = cutil.combine_distances([d1, d2])
                    # Distance corresponding to the best assignment of standards to each other
                    d_norm = np.min(normalizing_distances)

                    # Get the distance for every pair of network output and standards
                    distances = np.full((len(network_outputs), len(comparison_sounds)), np.nan)
                    for net_idx, net_gram in enumerate(network_outputs):
                        for comp_idx, comp_gram in enumerate(comparison_sounds):
                            mask = comp_gram > comp_gram.min()
                            distances[net_idx, comp_idx] = distance_metric(comp_gram, net_gram, mask)

                    # Find best network pair to match to mix, silence (cols 2 and 3), tone absent
                    noise_only_distance = np.full((len(network_outputs), len(network_outputs)), np.nan)
                    for net_idx_1 in range(len(network_outputs)):
                        for net_idx_2 in range(len(network_outputs)):
                            if net_idx_1 == net_idx_2:
                                noise_only_distance[net_idx_1, net_idx_2] = np.inf
                            else:
                                noise_only_distance[net_idx_1, net_idx_2] = cutil.combine_distances([distances[net_idx_1,2], distances[net_idx_2,3]])
                    # distance to noise only explanation, for outputs that best match noise only
                    d_nonly = np.min(noise_only_distance)

                    # Find best network pair to match to Noise, tone (cols 0 and 1), tone detected
                    noise_plus_tone_distance = np.full((len(network_outputs), len(network_outputs)), np.nan)
                    for net_idx_1 in range(len(network_outputs)):
                        for net_idx_2 in range(len(network_outputs)):
                            if net_idx_1 == net_idx_2:
                                noise_plus_tone_distance[net_idx_1, net_idx_2] = np.inf
                            else:
                                noise_plus_tone_distance[net_idx_1, net_idx_2] = cutil.combine_distances([distances[net_idx_1,0], distances[net_idx_2,1]])
                    # distance to noise+tone explanation, for outputs that best match noise+tone
                    d_nt = np.min(noise_plus_tone_distance)

                    # when J > 0, N+T is better. when J < 0 noise only is better
                    judgments[condition][bw][tone_level].append(
                        (d_nonly - d_nt)/d_norm
                        )

    return judgments


def figure1(sound_group, network, expt_name=None):

    # Load stimulus generation settings
    soundpath = os.path.join(os.environ["sound_dir"], sound_group, "/")
    seeds = range(10)
    settings = np.load(soundpath + "cmr_expt1_settings_seed0.npy", allow_pickle=True).item()
    settings["bandwidths"] = sorted(settings["bandwidths"])

    # Get network results
    print("Getting network results...")
    net_results = {}
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    for condition in conditions:
        net_results[condition] = {}
        for bw in settings["bandwidths"]:
            net_results[condition][bw] = {}
            for tone_level in settings["tone_levels"]:
                net_results[condition][bw][tone_level] = {}
                for seed in seeds:
                    # Get premixture sounds
                    sound_name = f"{condition}_tone_bw{bw}_toneLevel{tone_level}_seed{seed}-0"
                    network_outputs = []
                    fns = glob(netpath + f"{sound_name}_*_estimate.wav")
                    for f in fns:
                        network_output, net_sr = sf.read(f)
                        network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
                    # Always have at least two outputs
                    if len(network_outputs) == 1:
                        network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
                    net_results[condition][bw][tone_level][seed] = network_outputs

    # Get judgments
    comparison_dict = generate_comparison_sounds(sound_group, settings, seeds, network, resample_rate=None if "sequential-gen-model" in network else net_sr)
    judgments = get_distances(net_results, settings, comparison_dict, seeds)
    print(judgments)
    np.save(os.path.join(netpath, "raw_judgments.npy"), judgments)

    # Get thresholds
    tl = np.array(settings["tone_levels"])
    boundaries_to_save = {}
    for c_idx, condition in enumerate(conditions):
        for bw_idx, bw in enumerate(settings["bandwidths"]):
            boundaries = []
            for seed in seeds:
                y = np.array([
                        judgments[condition][bw][tone_level][int(seed)] for tone_level in settings["tone_levels"]
                    ])
                if np.all(y < 0):
                    boundary = max(settings["tone_levels"]) + 5
                else:
                    boundary = 0.5*(tl[(y <= 0).argmin()] + tl[(y >= 0).argmax()])
                boundaries.append(boundary)

    print(os.path.join(netpath, "result.png"))
    plt.savefig(os.path.join(netpath, "result.png"))
    np.save(os.path.join(netpath, "result.npy"), boundaries_to_save)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound_group", type=str, help="which sounds do you want to analyze?")
    parser.add_argument("--expt_name", default=None, type=str)
    args = parser.parse_args()
    model_results = figure1(args.sound_group, args.network, expt_name=args.expt_name)
