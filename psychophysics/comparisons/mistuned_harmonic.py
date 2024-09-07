import os
import numpy as np
import scipy.signal
from glob import glob
import soundfile as sf
import matplotlib.pyplot as plt

from util import context
import psychophysics.generation.mistuned_harmonic as gen
import psychophysics.comparisons.cutil as cutil


def generate_comparison_sounds(settings, sound_group, network, resample_rate=None):
    """Comparison sounds to convert network soundwaves into judgments"""

    sr = 20000
    context(audio_sr=sr, rms_ref=1e-6)
    all_components_dict, _ = gen.expt(
        "", overwrite=settings, save_wav=False, return_all_components=True
        )

    comparison_dicts = {}
    mistunedLevel = settings["mistuned_level"]
    for i in range(len(settings["fundamentals"])):
        f0 = settings["fundamentals"][i]
        comparison_dicts[f0] = {}
        for d in settings["durations"]:
            comparison_dicts[f0][d] = {}
            for mistunedIdx in settings["mistuned_idxs"]:
                comparison_dicts[f0][d][mistunedIdx] = {}
                for mtP in settings["mistuned_percents"]:
                    comparison_dicts[f0][d][mistunedIdx][mtP] = []
                    fn = f'f0{f0}_dur{d}_harm{mistunedIdx}_p{mtP}_dB{mistunedLevel}'
                    (mixture, mistuned_tone_only, harmonic_only) = all_components_dict[fn]
                    for sound in [mistuned_tone_only, harmonic_only, mixture, np.zeros(mixture.shape)]:
                        if "sequential-gen-model" in network:
                            # Account for pad from amortized neural network duration requirement
                            seconds_per_frame = 0.005
                            if len(sound) % int(np.round(seconds_per_frame*sr)) != 0:
                                N = int(np.round(seconds_per_frame*sr))
                                diff = int(np.ceil(len(sound) / N) * N) - len(sound)
                                sound = np.concatenate((sound, np.zeros((diff,))))
                        if resample_rate is not None:
                            sound = scipy.signal.resample_poly(sound, resample_rate, sr)
                            comparison_dicts[f0][d][mistunedIdx][mtP].append(cutil.make_gammatonegram(sound, resample_rate))
                        else:
                            comparison_dicts[f0][d][mistunedIdx][mtP].append(cutil.make_gammatonegram(sound, sr))

    return comparison_dicts

def get_distances(net_results, settings, comparison_dict):
    """ Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")
    judgments = {}
    for i in range(len(settings["fundamentals"])):
        f0 = settings["fundamentals"][i]
        judgments[f0] = {}
        for d in settings["durations"]:
            judgments[f0][d] = {}
            for mistunedIdx in settings["mistuned_idxs"]:
                judgments[f0][d][mistunedIdx] = {}
                for mtP in settings["mistuned_percents"]:
                    judgments[f0][d][mistunedIdx][mtP] = []
                    network_outputs = net_results[f0][d][mistunedIdx][mtP]
                    comparison_sounds = comparison_dict[f0][d][mistunedIdx][mtP] # tone, harmonic, mix, silence
                    mix_gram = comparison_dict[f0][d][mistunedIdx][mtP][2]
                    silence_gram = comparison_dict[f0][d][mistunedIdx][mtP][3]

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
                                # Compare norm_gram_1 to tone
                                d1 = distance_metric(comparison_sounds[0], norm_gram_1, comparison_sounds[0]>20)
                                # Compare norm_gram_2 to harmonic
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
                    harmonic_only_distance = np.full((len(network_outputs), len(network_outputs)), np.nan)
                    for net_idx_1 in range(len(network_outputs)):
                        for net_idx_2 in range(len(network_outputs)):
                            if net_idx_1 == net_idx_2:
                                harmonic_only_distance[net_idx_1, net_idx_2] = np.inf
                            else:
                                harmonic_only_distance[net_idx_1, net_idx_2] = cutil.combine_distances([distances[net_idx_1,2], distances[net_idx_2,3]])
                    d_absent = np.min(harmonic_only_distance)

                    # Find best network pair to match to harmonic, tone (cols 0 and 1), tone detected
                    harmonic_plus_tone_distance = np.full((len(network_outputs), len(network_outputs)), np.nan)
                    for net_idx_1 in range(len(network_outputs)):
                        for net_idx_2 in range(len(network_outputs)):
                            if net_idx_1 == net_idx_2:
                                harmonic_plus_tone_distance[net_idx_1, net_idx_2] = np.inf
                            else:
                                harmonic_plus_tone_distance[net_idx_1, net_idx_2] = cutil.combine_distances([distances[net_idx_1,0], distances[net_idx_2,1]])
                    d_present = np.min(harmonic_plus_tone_distance)

                    # when J > 0, H+T is better. when J < 0 harmonic only is better
                    judgments[f0][d][mistunedIdx][mtP].append((d_absent - d_present)/d_norm)

    return judgments


def expt(sound_group, network, expt_name=None):

    # Load stimulus generation settings
    soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
    settings = np.load(os.path.join(soundpath, "mh_expt_settings.npy"), allow_pickle=True).item()
    settings["mistuned_percents"] = sorted(settings["mistuned_percents"])
    mistunedLevel = settings["mistuned_level"]

    # Get network results
    print("Getting network results...")
    net_results = {}
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    net_results = {}
    for i in range(len(settings["fundamentals"])):
        f0 = settings["fundamentals"][i]
        net_results[f0] = {}
        for d in settings["durations"]:
            net_results[f0][d] = {}
            for mistunedIdx in settings["mistuned_idxs"]:
                net_results[f0][d][mistunedIdx] = {}
                for mtP in settings["mistuned_percents"]:
                    fn = f'f0{f0}_dur{d}_harm{mistunedIdx}_p{mtP}_dB{mistunedLevel}'
                    network_outputs = []
                    fns = glob(netpath + f"{fn}_*_estimate.wav")
                    for f in fns:
                        network_output, net_sr = sf.read(f)
                        network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
                    # Make sure there's always at least two outputs
                    if len(network_outputs) == 1:
                        network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
                    net_results[f0][d][mistunedIdx][mtP] = network_outputs

    # Get judgments
    comparison_dict = generate_comparison_sounds(
        settings, sound_group, network, resample_rate=None if "sequential-gen-model" in network else net_sr
        )
    judgments = get_distances(net_results, settings ,comparison_dict)
    print(judgments)
    np.save(os.path.join(netpath, "raw_judgments.npy"), judgments)

    # Get thresholds
    boundaries = {}
    mps = np.array(settings["mistuned_percents"])
    for i in range(len(settings["fundamentals"])):
        f0 = settings["fundamentals"][i]
        boundaries[f0] = {}
        for duration_idx, d in enumerate(settings["durations"]):
            for mistunedIdx in settings["mistuned_idxs"]:
                y = np.array([judgments[f0][d][mistunedIdx][_mp] for _mp in settings["mistuned_percents"]])
                if np.all(y < 0):
                    boundary = max(settings["mistuned_percents"]) + 5
                elif np.all(y > 0):
                    boundary = min(settings["mistuned_percents"])
                else:
                    boundary = 0.5*(mps[(y <= 0).argmin()] + mps[(y >= 0).argmax()])
                boundaries[(f0, mistunedIdx)] = boundary  # There should just be one duration

    print(os.path.join(netpath, "lines.png"))
    plt.savefig(os.path.join(netpath, "lines.png"))
    np.save(os.path.join(netpath, "result.npy"), boundaries) 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound-group", type=str,
                        help="which sounds do you want to analyze?")
    parser.add_argument("--expt-name", default=None, type=str,
                        help="which inferences do you want to analyze?")
    args = parser.parse_args()
    model_results = expt(args.sound_group, args.network, expt_name=args.expt_name)
