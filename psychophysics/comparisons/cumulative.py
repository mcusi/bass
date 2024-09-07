import os
import numpy as np
from glob import glob
import scipy.signal
import soundfile as sf
import matplotlib.cm
import matplotlib.pyplot as plt
from copy import copy, deepcopy

from util import context
import psychophysics.comparisons.cutil as cutil
from psychophysics.analysis.tone_sequence_summary import plot_cumulative
import psychophysics.generation.vanNoorden1975 as gen


def generate_comparison_sounds(network, resample_rate=None):
    """Comparison sounds to convert network soundwaves into judgments"""

    with context(audio_sr=20000, rms_ref=1e-6):
        basic_cumul = {
            "cf": 500,
            "dt": [125],
            "df": [4, 8],
            "level": 55,
            "n_reps": [1, 2, 3, 4, 5, 6]
        }  # See psychophysics/generation/tone_sequences.py
        galloping_dict, gen_sr = gen.expt1("", overwrite=basic_cumul, save_wav=False)
        iso_cumul_B = deepcopy(basic_cumul)
        iso_cumul_B["levelB"] = 0
        isochronousA_dict, gen_sr = gen.expt1("", overwrite=iso_cumul_B, save_wav=False)
        iso_cumul_A = deepcopy(basic_cumul)
        iso_cumul_A["levelA"] = 0
        isochronousB_dict, gen_sr = gen.expt1("", overwrite=iso_cumul_A, save_wav=False)
    
    for d in [galloping_dict, isochronousA_dict, isochronousB_dict]:
        for k in d.keys():
            if "sequential-gen-model" in network:
                # Account for amortized neural network modifying duration
                seconds_per_frame = 0.005
                if len(d[k]) % int(np.round(seconds_per_frame*gen_sr)) != 0:
                    N = int(np.round(seconds_per_frame*gen_sr))
                    diff = int(np.ceil(len(d[k]) / N) * N) - len(d[k])
                    d[k] = np.concatenate((d[k], np.zeros((diff,))))
                # Need to trim these sounds to four seconds because the networks can only go up to that much - but should not be relevant at n_reps = 6
                d[k] = d[k][:int(4*gen_sr)]
            if resample_rate is not None:
                wav_rs = scipy.signal.resample_poly(d[k], resample_rate, gen_sr)
                d[k] = cutil.make_gammatonegram(wav_rs, resample_rate)
            else:
                # print(k, d[k].shape)
                d[k] = cutil.make_gammatonegram(d[k], gen_sr)

    return galloping_dict, isochronousA_dict, isochronousB_dict


def compare(net_results, galloping_dict, isochronousA_dict, isochronousB_dict, dfs, dts, n_reps):
    """Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")

    judgments = np.zeros((len(dfs), len(dts), len(n_reps), 2))
    for nr_idx, n_rep in enumerate(n_reps):
        print(n_rep)
        for df_idx, df in enumerate(dfs):
            for dt_idx, dt in enumerate(dts):
                network_outputs = net_results[(df, dt, n_rep)]
                comparison_gtgs = {}
                comparison_gtgs["g"] = galloping_dict[f"df{df}_dt{dt}_rep{n_rep}.wav"]
                comparison_gtgs["silence"] = 20 + 0*galloping_dict[f"df{df}_dt{dt}_rep4.wav"]
                comparison_gtgs["iA"] = isochronousA_dict[f"df{df}_dt{dt}_rep{n_rep}.wav"]
                comparison_gtgs["iB"] = isochronousB_dict[f"df{df}_dt{dt}_rep{n_rep}.wav"]

                # Get the distance for every pair of network output and standards
                distances = {}
                for k, v in comparison_gtgs.items():
                    distances[k] = []
                    mask = v > v.min()
                    for network_output in network_outputs:
                        distances[k].append(distance_metric(v, network_output, mask))

                # Get the distance between every pair of standards
                # This provides the denominator for the preference function
                normalizing_distances = np.full((2, 2), np.nan)
                for n_idx_1 in range(2):
                    norm_gram_1 = comparison_gtgs["g"] if n_idx_1 == 0 else comparison_gtgs["silence"]
                    for n_idx_2 in range(2):
                        norm_gram_2 = comparison_gtgs["g"] if n_idx_2 == 0 else comparison_gtgs["silence"]
                        if n_idx_1 == n_idx_2:
                            normalizing_distances[n_idx_1, n_idx_2] = np.inf
                        else:
                            d1=distance_metric(comparison_gtgs["iA"], norm_gram_1, comparison_gtgs["iA"]>20)
                            d2=distance_metric(comparison_gtgs["iB"], norm_gram_2, comparison_gtgs["iB"]>20)
                            normalizing_distances[n_idx_1, n_idx_2] = cutil.combine_distances([d1, d2])
                # Distance corresponding to the best assignment of standards to each other
                d_norm = np.min(normalizing_distances)

                distance_matrices = {}
                # Find best distance to galloping
                distance_matrices["galloping"] = np.zeros((len(network_outputs), len(network_outputs)))
                for a_idx, a_distance in enumerate(distances["g"]):
                    for b_idx, b_distance in enumerate(distances["silence"]):
                        if a_idx == b_idx and len(network_outputs)>1:
                            distance_matrices["galloping"][a_idx, b_idx] = np.inf
                        else:
                            distance_matrices["galloping"][a_idx, b_idx] = cutil.combine_distances([a_distance, b_distance])
                # distance to galloping explanation, for outputs that best match galloping
                d_gallop = np.min(distance_matrices["galloping"])

                # Find best distance to isochronous
                distance_matrices["isochronous"] = np.zeros((len(network_outputs), len(network_outputs)))
                for a_idx, a_distance in enumerate(distances["iA"]):
                    for b_idx, b_distance in enumerate(distances["iB"]):
                        if a_idx == b_idx:
                            distance_matrices["isochronous"][a_idx, b_idx] = np.inf
                        else:
                            distance_matrices["isochronous"][a_idx, b_idx] = cutil.combine_distances([a_distance, b_distance])
                # shortest distance to isochonrous, for outputs that best match isochronous 
                d_iso = np.min(distance_matrices["isochronous"])

                judgments[df_idx, dt_idx, nr_idx, 1] = d_iso/d_norm
                judgments[df_idx, dt_idx, nr_idx, 0] = d_gallop/d_norm

    return judgments


def main(sound_group, network, expt_name):

    # Load in network results
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    net_results = {}
    n_reps = [1, 2, 3, 4, 5, 6]
    dts = [125]
    dfs = [4, 8]
    for n_rep in n_reps:
        for df in dfs:
            for dt in dts:
                fns = glob(netpath + f"df{df}_dt{dt}_rep{n_rep}_*s*_estimate.wav")
                network_outputs = []
                for fn in fns:
                    network_output, net_sr = sf.read(fn)
                    network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
                if len(network_outputs) == 1:
                    network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
                net_results[(df, dt, n_rep)] = network_outputs
    
    # Get comparison stimuli and resample if necessary
    galloping_dict, isochronousA_dict, isochronousB_dict = generate_comparison_sounds(network, resample_rate=net_sr if network != "sequential-gen-model" else None)

    # Compare in gammatonegram space
    judgments = compare(net_results, galloping_dict, isochronousA_dict, isochronousB_dict, dfs, dts, n_reps)
    # For distances, smaller is better. For log odds, bigger is better.
    # Use negative because plot_cumulative will go J[1]-J[0] == J[iso_distance] - J[gallop_distance]
    print(os.path.join(netpath, 'result_cumul.png'))
    print(judgments)
    plot_cumulative(
        -np.transpose(judgments, (2, 1, 0, 3))[:, :, :, :, None],
        n_reps, sound_group, network, None,
        savepath=os.path.join(netpath, 'result_cumul.png')
        )
    model_results = {"tfr": [], "d": []}
    for nr_idx, n_rep in enumerate(n_reps):
        for df_idx, df in enumerate(dfs):
            for dt_idx, dt in enumerate(dts):
                model_results["tfr"].append((dt, df, n_rep))
                model_results["d"].append(
                    judgments[df_idx, dt_idx, nr_idx, 0] - judgments[df_idx, dt_idx, nr_idx, 1]
                    )
    np.save(os.path.join(netpath, "result_cumul.npy"), model_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound-group", type=str,
                        help="which sounds do you want to analyze?")
    parser.add_argument("--expt-name", default=None, type=str,
                        help="which inferences to analyze?")
    args = parser.parse_args()
    print("Plotting figure...")
    main(args.sound_group, args.network, expt_name=args.expt_name)
