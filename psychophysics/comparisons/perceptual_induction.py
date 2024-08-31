import os
import numpy as np
from glob import glob
import soundfile as sf
import scipy.signal
import torch

from util import context
import psychophysics.comparisons.cutil as cutil
import psychophysics.generation.perceptual_induction as gen
import psychophysics.generation.basic as basic
import psychophysics.generation.tones as tones


def generate_comparison_sounds(settings, sound_group, network, resample_rate=None):
    """ Comparison sounds to convert network soundwaves into judgments
        Generate masker noises on their own, separated tones,
        and continuous tones on their own
    """

    sr = settings["audio_sr"]
    context(audio_sr=sr, rms_ref=1e-6)

    # Create mixture
    mixture_settings = settings.copy()
    mixtures_dict, _ = gen.expt3(sound_group, overwrite=mixture_settings, save_wav=False)

    # Create noises only
    noise_settings = settings.copy()
    noise_settings["tone_levels"] = [0.0]
    noises_dict, _ = gen.expt3(sound_group, overwrite=noise_settings, save_wav=False)

    # Create discontinuous tones (for masking and continuity)
    dc_tone_settings = settings.copy()
    dc_tone_settings["maskerLevel"] = 0.0
    dc_tone_dict, _ = gen.expt3(sound_group, overwrite=dc_tone_settings, save_wav=False)

    # Create continuous tones (for continuity only)
    c_tone_dict = {}
    durations = {"ramp": settings["rampDuration"]}
    for f in settings["tone_freqs"]:
        for level in settings["tone_levels"]:
            target = basic.raisedCosineRamps(
                # TODO: Durations are hard-coded
                tones.pure(
                    1.785, f, level=level, phase=np.random.rand()*2*np.pi
                    ),
                durations['ramp']
                )
            fn = 'continuity_f{:04}_l{:03}'.format(f, level)
            stimulus = basic.addSilence(target, [0.480, 0.484])
            c_tone_dict[fn] = stimulus

    for d_idx, d in enumerate([noises_dict, dc_tone_dict, c_tone_dict, mixtures_dict]):
        for k in d.keys():
            if "sequential-gen-model" in network:
                # Account for pad from amortized neural network duration requirement
                seconds_per_frame = 0.005
                if len(d[k]) % int(np.round(seconds_per_frame*sr)) != 0:
                    N = int(np.round(seconds_per_frame*sr))
                    diff = int(np.ceil(len(d[k]) / N) * N) - len(d[k])
                    d[k] = np.concatenate((d[k], np.zeros((diff,))))
            if resample_rate is not None:
                d[k] = scipy.signal.resample_poly(d[k], resample_rate, sr)
                d[k] = cutil.make_gammatonegram(d[k], resample_rate)
            else:
                d[k] = cutil.make_gammatonegram(d[k], sr)

    return noises_dict, dc_tone_dict, c_tone_dict, mixtures_dict


def compare(net_results, noises_dict, dc_tone_dict, c_tone_dict, mixtures_dict, settings):
    """ Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")

    judgments = {}
    for condition in ["masking", "continuity"]:
        judgments[condition] = {}
        for f in settings["tone_freqs"]:
            judgments[condition][f] = {}
            for lv in settings["tone_levels"]:
                # Get correct comparison gammatonegrams
                network_outputs = net_results[condition][f][lv]
                comparison_gtgs = {}
                # Noise only
                noise_to_compare = [k for k in noises_dict.keys() if condition in k and f"_f{f:04d}_" in k][0]
                comparison_gtgs["n"] = noises_dict[noise_to_compare]
                # Mixture, tone + noise
                noise_to_compare = [k for k in mixtures_dict.keys() if condition in k and f"_f{f:04d}_" in k and f"_l{lv:03d}" in k][0]
                comparison_gtgs["m"] = mixtures_dict[noise_to_compare]
                # In masking: tone only present; in continuity: discontinuous tone
                dc_tone_to_compare = f"{condition}_f{f:04d}_l{lv:03d}"
                comparison_gtgs["dc"] = dc_tone_dict[dc_tone_to_compare]
                if condition == "continuity":
                    # In continuity: continuous tone
                    c_tone_to_compare = f"{condition}_f{f:04d}_l{lv:03d}"
                    comparison_gtgs["c"] = c_tone_dict[c_tone_to_compare]
                comparison_gtgs["silence"] = 0*comparison_gtgs["m"] + 20.

                # Find distance between network outputs and comparison stimuli
                distances = {}
                for k, v in comparison_gtgs.items():
                    distances[k] = []
                    mask = v > v.min()
                    for network_output in network_outputs:
                        d = distance_metric(v, network_output, mask)
                        distances[k].append(d)

                if condition == "masking":

                    # Get the distance between every pair of standards
                    # This provides the denominator for the preference function
                    # Set 1: mixture and silence
                    # Set 2: noise and tone
                    normalizing_distances = np.full((2, 2), np.nan)
                    mix_gram = comparison_gtgs["m"]
                    for n_idx_1 in range(2):
                        norm_gram_1 = mix_gram if n_idx_1 == 0  else comparison_gtgs["silence"]
                        for n_idx_2 in range(2):
                            norm_gram_2 = mix_gram if n_idx_2 == 0  else comparison_gtgs["silence"]
                            if n_idx_1 == n_idx_2:
                                normalizing_distances[n_idx_1, n_idx_2] = np.inf
                            else:
                                # Compare to noise
                                d1=distance_metric(comparison_gtgs["n"], norm_gram_1, comparison_gtgs["n"]>20)
                                # Compare to tone
                                d2=distance_metric(comparison_gtgs["dc"], norm_gram_2, comparison_gtgs["dc"]>20)
                                normalizing_distances[n_idx_1, n_idx_2] = cutil.combine_distances([d1, d2])
                    d_norm = np.min(normalizing_distances)

                    # Find best distance to each answer
                    distance_matrices = {}
                    for answer in ["n+t", "n"]:  # tone present, tone absent
                        distance_matrices[answer] = np.zeros((len(network_outputs), len(network_outputs)))
                        for n_idx, n_distance in enumerate(distances["n"]):
                            for dc_idx, dc_distance in enumerate(distances["dc"]):
                                if n_idx == dc_idx and len(network_outputs) > 1:
                                    distance_matrices[answer][n_idx, dc_idx] = np.inf
                                elif answer == "n":
                                    distance_matrices[answer][n_idx, dc_idx] = cutil.combine_distances([distances["m"][n_idx], distances["silence"][dc_idx]])
                                elif answer == "n+t":
                                    distance_matrices[answer][n_idx, dc_idx] = cutil.combine_distances([n_distance, dc_distance])
                    # distance to each explanation, for outputs that best match that explanation
                    d_present = np.min(distance_matrices["n+t"])
                    d_absent = np.min(distance_matrices["n"])
                    # J < 0 means n is better, J > 0 means n+t is better
                    judgments[condition][f][lv] = (d_absent - d_present)/d_norm

                elif condition == "continuity":
                    # Get the distance between every pair of standards
                    # This provides the denominator for the preference function
                    # Set 1: noise and discontinuous tone
                    # Set 2: noise and continuous tone
                    normalizing_distances = np.full((2, 2), np.nan)
                    mix_gram = comparison_gtgs["m"]
                    for n_idx_1 in range(2):
                        norm_gram_1 = comparison_gtgs["n"] if n_idx_1 == 0 else comparison_gtgs["dc"]
                        for n_idx_2 in range(2):
                            norm_gram_2 = comparison_gtgs["n"] if n_idx_2 == 0 else comparison_gtgs["dc"]
                            if n_idx_1 == n_idx_2:
                                normalizing_distances[n_idx_1, n_idx_2] = np.inf
                            else:
                                # Compare to noise
                                d1=distance_metric(comparison_gtgs["n"], norm_gram_1, comparison_gtgs["n"]>20)
                                # Compare to continuous tone
                                d2=distance_metric(comparison_gtgs["c"], norm_gram_2, comparison_gtgs["c"]>20)
                                normalizing_distances[n_idx_1, n_idx_2] = cutil.combine_distances([d1, d2])
                    d_norm = np.min(normalizing_distances)

                    # Select whether you should use the continuous or discontinuous option
                    # Find best distance to each answer
                    distance_matrices = {}
                    for answer in ["c", "dc"]:  # continuous, discontinuous
                        distance_matrices[answer] = np.zeros((len(network_outputs), len(network_outputs)))
                        for n_idx, n_distance in enumerate(distances["n"]):
                            for tone_idx, tone_distance in enumerate(distances[answer]):
                                if n_idx == tone_idx:
                                    distance_matrices[answer][n_idx, tone_idx] = np.inf
                                else:
                                    distance_matrices[answer][n_idx, tone_idx] = cutil.combine_distances([n_distance, tone_distance])
                    # distance to each explanation, for outputs that best match that explanation
                    d_continuous = np.min(distance_matrices["c"])
                    d_discontinuous = np.min(distance_matrices["dc"])
                    # J < 0 means C is better, J > 0 means dc is better
                    judgments[condition][f][lv] = (d_continuous - d_discontinuous)/d_norm
                    """
                    Three possibilities, since distances >= 0:
                    `   `
                    ------
                    C   dc --> same values, equally far from zero. J = 0

                    `
                    `
                    ------
                    C  dc --> dc is closer to zero == less distance == better explanation. J > 0

                    `
                    `
                    -----
                    C  dc --> C is closer to zero, J < 0
                    """

    return judgments


def expt3(sound_group, network, expt_name=None):

    settings_fn = os.environ["sound_dir"] + sound_group + "/pi_expt3_settings_seed0.npy"
    settings = np.load(settings_fn, allow_pickle=True).item()
    settings["tone_levels"] = sorted(settings["tone_levels"])
    settings["tone_freqs"] = sorted(settings["tone_freqs"])
    audio_sr = settings["audio_sr"]

    # Load in network results 
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    net_results = {}
    for condition in ["masking", "continuity"]:
        net_results[condition] = {}
        for f in settings["tone_freqs"]:
            net_results[condition][f] = {}
            for lv in settings["tone_levels"]:
                fns = glob(netpath + f"{condition}_f{f:04d}_l{lv:03d}_s*_estimate.wav")
                network_outputs = []
                for fn in fns:
                    network_output, net_sr = sf.read(fn)
                    network_outputs.append(
                        cutil.make_gammatonegram(network_output, net_sr).detach().numpy()
                        )
                # Make sure there's always at least two network outputs
                if len(network_outputs) == 1:
                    network_outputs.append(
                        cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr).detach().numpy()
                        )
                net_results[condition][f][lv] = network_outputs

    # Get comparison stimuli and resample if necessary
    noises_dict, dc_tone_dict, c_tone_dict, mixtures_dict = generate_comparison_sounds(
        settings, sound_group, network, resample_rate=net_sr if net_sr != audio_sr else None
        )
    # Compare in gammatonegram space
    judgments = compare(net_results, noises_dict, dc_tone_dict, c_tone_dict, mixtures_dict, settings)
    np.save(os.path.join(netpath, "distances.npy"), judgments)
    print(judgments)

    # Get thresholds
    boundaries = {"masking": [], "continuity": []}
    tlv = settings["tone_levels"]
    for f in settings["tone_freqs"]:
        # Masking
        y = np.array([judgments["masking"][f][lv] for lv in tlv])
        if np.all(y > 0):
            boundary = tlv[0] - 4
        elif np.all(y < 0):
            boundary = tlv[-1] + 4
        else:
            boundary = 0.5*(tlv[(y >= 0).argmax()] + tlv[(y <= 0).argmin()])
        boundaries["masking"].append(boundary)
        # Continuity
        y = np.array([
            judgments["continuity"][f][lv] for lv in tlv
            ])
        if np.all(y > 0):
            boundary = tlv[0] - 4
        elif np.all(y < 0):
            boundary = tlv[-1] + 4
        else:
            boundary = 0.5*(tlv[(-y <= 0).argmax()] + tlv[(-y >= 0).argmin()])
        boundaries["continuity"].append(boundary)

    print(os.path.join(netpath, "result.npy"))
    np.save(os.path.join(netpath, "result.npy"), boundaries)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound_group", type=str, help="which sounds do you want to analyze?")
    parser.add_argument("--expt_name", default=None, type=str, help="which sounds do you want to analyze?")
    args = parser.parse_args()
    expt3(args.sound_group, args.network, expt_name=args.expt_name)
