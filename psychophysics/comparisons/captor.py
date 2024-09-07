import os
import numpy as np
from glob import glob
import scipy.signal
import soundfile as sf

import psychophysics.generation.BregmanRudnicky1975 as gen
from psychophysics.analysis.tone_sequence_summary import plot_captor
import psychophysics.comparisons.cutil as cutil
from util import context


def generate_comparison_sounds(network, resample_rate=None):
    """ Comparison sounds to convert network soundwaves into judgments
        We need to generate three kinds of standard sets
        1. (captors + distractors) + (targets)
        2. (target + distractors) + (captors)
        3. (target) + (distractors) + (captors)
    """

    context(audio_sr=20000, rms_ref=1e-6)
    # Captors only
    overwrite_captors = {"level_a": -100, "level_b": -100, "level_distractor":-100}
    captors_sounds, gen_sr = gen.simple_expt1("", overwrite=overwrite_captors, save_wav=False)
    # Distractors only
    overwrite_distractors = {"level_a": -100, "level_b": -100, "level_distractor": 60, "level_captors": [-100, -100, -100]}
    distractor_sounds, gen_sr = gen.simple_expt1("", overwrite=overwrite_distractors, save_wav=False)
    # Target only
    overwrite_target = { "level_distractor":-100, "level_captors": [-100, -100, -100]}
    target_sounds, gen_sr = gen.simple_expt1("", overwrite=overwrite_target, save_wav=False)

    if resample_rate is not None:
        for d in [captors_sounds, distractor_sounds, target_sounds]:
            for k in d.keys():
                d[k] = scipy.signal.resample_poly(d[k], resample_rate, gen_sr)
    
    sounds_to_use = {}
    for k in ["compdown_f590.wav", "compdown_f1030.wav", "compdown_f1460.wav"]:
        # Leave out compdown_fnone because there are no captors
        sounds_to_use["captor_" + k[:-4]] = captors_sounds[k]
    sounds_to_use["distractor"] = distractor_sounds["compdown_fnone.wav"]
    sounds_to_use["target"] = target_sounds["compdown_fnone.wav"]

    for k in [_k for _k in sounds_to_use.keys() if "captor" in _k]:
        sounds_to_use["distractor+"+k] = sounds_to_use["distractor"] + sounds_to_use[k]
    sounds_to_use["distractor+target"] = sounds_to_use["distractor"] + sounds_to_use["target"]

    for k in sounds_to_use.keys():
        if "sequential-gen-model" in network:
            # Account for pad from amortized neural network duration requirement
            seconds_per_frame = 0.005
            if len(sounds_to_use[k]) % int(np.round(seconds_per_frame*gen_sr)) != 0:
                N = int(np.round(seconds_per_frame*gen_sr))
                diff = int(np.ceil(len(sounds_to_use[k]) / N) * N) - len(sounds_to_use[k])
                sounds_to_use[k] = np.concatenate((sounds_to_use[k], np.zeros((diff,))))
        sounds_to_use[k] = cutil.make_gammatonegram(sounds_to_use[k], resample_rate if network != "sequential-gen-model" else gen_sr)

    silence = 0*sounds_to_use["distractor+target"] + 20
    explanations = {
        # For none, there are no captors, but naming is easier if we include
        "none": {
                "(T+D)+(C)": [sounds_to_use["distractor+target"], silence],
                "(T)+(D)+(C)": [sounds_to_use["distractor"], sounds_to_use["target"]]
                },
        "590": {
            "(T+D)+(C)": [sounds_to_use["distractor+target"], sounds_to_use["captor_compdown_f590"], silence],
            "(T)+(D+C)": [sounds_to_use["target"], sounds_to_use["distractor+captor_compdown_f590"], silence],
            "(T)+(D)+(C)": [sounds_to_use["target"], sounds_to_use["distractor"], sounds_to_use["captor_compdown_f590"]]
            },
        "1030": {
            "(T+D)+(C)": [sounds_to_use["distractor+target"], sounds_to_use["captor_compdown_f1030"], silence],
            "(T)+(D+C)": [sounds_to_use["target"], sounds_to_use["distractor+captor_compdown_f1030"], silence],
            "(T)+(D)+(C)": [sounds_to_use["target"], sounds_to_use["distractor"], sounds_to_use["captor_compdown_f1030"]]
            },
        "1460": {
            "(T+D)+(C)": [sounds_to_use["distractor+target"], sounds_to_use["captor_compdown_f1460"], silence],
            "(T)+(D+C)": [sounds_to_use["target"], sounds_to_use["distractor+captor_compdown_f1460"], silence],
            "(T)+(D)+(C)": [sounds_to_use["target"], sounds_to_use["distractor"], sounds_to_use["captor_compdown_f1460"]]
            }
    }

    return explanations


def compare(net_results, explanations):
    """ Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")
    judgments = np.zeros((4, 3))
    for cond_idx, condition in enumerate(["none", "590", "1030", "1460"]):
        network_outputs = net_results[condition]
        comparison_gtgs = explanations[condition]

        # First get denominator for the preference measure
        # Distances of standard sets to themselves
        if condition == "none":
            normalizing_distances = np.full((2, 2), np.nan)
            for n_idx_1 in range(2):
                norm_gram_1 = comparison_gtgs['(T+D)+(C)'][0] if n_idx_1 == 0 else comparison_gtgs['(T+D)+(C)'][1]
                for n_idx_2 in range(2):
                    norm_gram_2 = comparison_gtgs['(T+D)+(C)'][0] if n_idx_2 == 0 else comparison_gtgs['(T+D)+(C)'][1]
                    if n_idx_1 == n_idx_2:
                        normalizing_distances[n_idx_1, n_idx_2] = np.inf
                    else:
                        d1 = distance_metric(comparison_gtgs['(T)+(D)+(C)'][0], norm_gram_1, comparison_gtgs['(T)+(D)+(C)'][0]>20)
                        d2 = distance_metric(comparison_gtgs['(T)+(D)+(C)'][1], norm_gram_2, comparison_gtgs['(T)+(D)+(C)'][1]>20)
                        normalizing_distances[n_idx_1, n_idx_2] = cutil.combine_distances([d1, d2])
        else:
            normalizing_distances = np.full((3, 3, 3, 2), np.nan)
            for batch_idx in range(2):
                cgs = comparison_gtgs["(T)+(D+C)"] if batch_idx == 0 else comparison_gtgs["(T)+(D)+(C)"]
                for n_idx_1 in range(3):
                    for n_idx_2 in range(3):
                        for n_idx_3 in range(3):
                            norm_gram_1 = cgs[n_idx_1]
                            norm_gram_2 = cgs[n_idx_2]
                            norm_gram_3 = cgs[n_idx_3]
                            if (n_idx_1 == n_idx_2) or (n_idx_1 == n_idx_3) or (n_idx_3 == n_idx_2):
                                normalizing_distances[n_idx_1, n_idx_2, n_idx_3, batch_idx] = np.inf
                            else:
                                d1 = distance_metric(comparison_gtgs["(T+D)+(C)"][0], norm_gram_1, comparison_gtgs["(T+D)+(C)"][0]>20)
                                d2 = distance_metric(comparison_gtgs["(T+D)+(C)"][1], norm_gram_2, comparison_gtgs["(T+D)+(C)"][1]>20)
                                d3 = distance_metric(comparison_gtgs["(T+D)+(C)"][2], norm_gram_3, comparison_gtgs["(T+D)+(C)"][2]>20)
                                normalizing_distances[n_idx_1, n_idx_2, n_idx_3, batch_idx] = cutil.combine_distances([d1, d2, d3])
        norm_dist = np.min(normalizing_distances)

        distance_matrices = {}
        for k in ["(T+D)+(C)", "(T)+(D+C)", "(T)+(D)+(C)"]:
            if k not in comparison_gtgs.keys():
                continue
            if len(network_outputs) < len(comparison_gtgs[k]):
                # If network only has two outputs, can't be (T)+(D)+(C)
                continue
            
            compare_to_explanations = np.zeros((len(network_outputs), len(comparison_gtgs[k])))
            for v_idx, v in enumerate(comparison_gtgs[k]):
                # Find distance of each network output to each sound in standard set
                mask = v > v.min()
                for n_idx, network_output in enumerate(network_outputs):
                    compare_to_explanations[n_idx, v_idx] = distance_metric(v, network_output, mask)

                # Find all the combinations of network outputs and standards
                if len(comparison_gtgs[k]) == 2:
                    distance_matrices[k] = np.zeros((len(network_outputs), len(network_outputs)))
                    for a_idx, a_distance in enumerate(compare_to_explanations[:, 0].tolist()):
                        for b_idx, b_distance in enumerate(compare_to_explanations[:, 1].tolist()):
                            if a_idx == b_idx:
                                distance_matrices[k][a_idx, b_idx] = np.inf
                            else:
                                distance_matrices[k][a_idx, b_idx] = cutil.combine_distances([a_distance, b_distance])
                elif len(comparison_gtgs[k])  == 3:
                    distance_matrices[k] = np.zeros((len(network_outputs), len(network_outputs), len(network_outputs)))
                    for a_idx, a_distance in enumerate(compare_to_explanations[:, 0].tolist()):
                        for b_idx, b_distance in enumerate(compare_to_explanations[:, 1].tolist()):
                            for c_idx, c_distance in enumerate(compare_to_explanations[:, 2].tolist()):
                                if (a_idx == b_idx) or (b_idx == c_idx) or (a_idx == c_idx):
                                    distance_matrices[k][a_idx, b_idx, c_idx] = np.inf
                                else:
                                    distance_matrices[k][a_idx, b_idx, c_idx] = cutil.combine_distances([a_distance, b_distance, c_distance])
                elif len(comparison_gtgs[k]) == 1:
                    distance_matrices[k] = compare_to_explanations[:, 0]

        # Find best matches and normalize. Later we will take the difference to get a preference
        # Target alone distance: "(T)+(D)+(C)" and "(T)+(D+C)"
        judgments[cond_idx, 1] = np.min(distance_matrices["(T)+(D)+(C)"])/norm_dist if "(T)+(D)+(C)" in distance_matrices.keys() else np.nan
        judgments[cond_idx, 2] = np.min(distance_matrices["(T)+(D+C)"])/norm_dist if "(T)+(D+C)" in distance_matrices.keys() else np.nan
        # Target with distractors distance
        judgments[cond_idx, 0] = np.min(distance_matrices["(T+D)+(C)"])/norm_dist

    print(judgments)
    return judgments


def main(sound_group, network, expt_name=None):

    # Load in network results
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    net_results = {}
    conditions = ["none", "590", "1030", "1460"]
    for condition in conditions:
        fns = glob(netpath + f"compdown_f{condition}_*s*_estimate.wav")
        network_outputs = []
        for fn in fns:
            network_output, net_sr = sf.read(fn)
            network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
        if len(network_outputs) == 1:
            network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
        if len(network_outputs) == 2:
            network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
        net_results[condition] = network_outputs

    # Get comparison stimuli and resample if necessary
    explanations = generate_comparison_sounds(network, resample_rate=net_sr if network != "sequential-gen-model" else None)

    # Compare in gammatonegram space
    judgments = compare(net_results, explanations)
    plot_captor(
        -judgments[:, :, None], conditions,
        sound_group, network,
        savepath=os.path.join(netpath, "result_captor.png")
        )
    print(os.path.join(netpath, "result_captor.png"))
    model_results = {"conditions": conditions, "d": []}
    for c_idx, condition in enumerate(conditions):
        model_results["d"].append(
            # 0="(T+D)+C", 1="(T)+(D)+(C)" or "(T)+(D+C)"
            judgments[c_idx, 0] - np.nanmin(judgments[c_idx, 1:])
            )
    np.save(os.path.join(netpath + "result_captor.npy"), model_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound_group", type=str,
                        help="which sounds do you want to analyze?")
    parser.add_argument("--expt_name", default=None, type=str,
                        help="which inferences to analyze?")
    args = parser.parse_args()
    main(args.sound_group, args.network, expt_name=args.expt_name)
