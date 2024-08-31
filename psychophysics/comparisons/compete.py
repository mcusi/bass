import os
import numpy as np
from glob import glob
import scipy.signal
import soundfile as sf

from util import context
import psychophysics.comparisons.cutil as cutil
import psychophysics.generation.bregman1978Competition as gen
from psychophysics.analysis.tone_sequence_summary import plot_compete_average


def generate_comparison_sounds(network, sound_fns, resample_rate=None):
    """ Comparison sounds to convert network soundwaves into judgments
        We need to generate different arrangements of A, B, X and Y
    """

    # Generate separate tones
    audio_sr = 20000
    context(audio_sr=audio_sr, rms_ref=1e-6)
    n_repetitions = 4
    silent_tone = -20
    As = gen.simple_expt2("", overwrite={"n_repetitions":n_repetitions,"abxy_levels":[None,silent_tone,silent_tone,silent_tone]}, save_wav=False)
    Bs = gen.simple_expt2("", overwrite={"n_repetitions":n_repetitions,"abxy_levels":[silent_tone,None,silent_tone,silent_tone]}, save_wav=False)
    Xs = gen.simple_expt2("", overwrite={"n_repetitions":n_repetitions,"abxy_levels":[silent_tone,silent_tone,None,silent_tone]}, save_wav=False)
    Ys = gen.simple_expt2("", overwrite={"n_repetitions":n_repetitions,"abxy_levels":[silent_tone,silent_tone,silent_tone,None]}, save_wav=False)

    # Create all the partitions that correspond to the two hypotheses
    # See competition_design in psychophysics.hypotheses.tone_sequences
    from psychophysics.hypotheses.tone_sequences import partition
    tone_idxs = list(range(4))
    compete_hypotheses = []
    for n, p in enumerate(partition(tone_idxs), 1):
        org = []
        for i in range(4):
            where_i = np.where([(i in _p) for _p in p])[0][0]
            org.append(where_i)
        fn = "".join([str(o) for o in org])
        compete_hypotheses.append(fn)
    compete_hypotheses = [fn for fn in compete_hypotheses if fn not in ["0123","0111","1011","0001","1101"]]
    # Reorder, just so isolate will be first!
    compete_hypotheses = ['0011', '0012'] + [fn for fn in compete_hypotheses if fn not in ['0011', '0012']]

    audio_comparison_dict = {}
    for sound_fn in sound_fns:
        audio_comparison_dict[sound_fn] = {}
        A = As[sound_fn]
        B = Bs[sound_fn]
        X = Xs[sound_fn]
        Y = Ys[sound_fn]
        for H_key in compete_hypotheses:
            n_sources = len(list(set(H_key)))
            audio_comparison_dict[sound_fn][H_key] = [np.zeros(A.shape) for i in range(3)]  # include silence
            for letter_idx, letter in enumerate([A, B, X, Y]):
                audio_comparison_dict[sound_fn][H_key][int(H_key[letter_idx])] += letter

    cgram_comparison_dict = {}
    for sound_fn, sound_dict in audio_comparison_dict.items():
        cgram_comparison_dict[sound_fn] = {}
        for H_key, list_of_sources in sound_dict.items():
            cgram_comparison_dict[sound_fn][H_key] = []
            for source in list_of_sources:
                if "sequential-gen-model" in network:
                    # Account for pad from amortized neural network duration requirement
                    seconds_per_frame = 0.005
                    if len(source) % int(np.round(seconds_per_frame*audio_sr)) != 0:
                        N = int(np.round(seconds_per_frame*audio_sr))
                        diff = int(np.ceil(len(source) / N) * N) - len(source)
                        source = np.concatenate((source, np.zeros((diff,))))
                if resample_rate is not None:
                    wav_rs = scipy.signal.resample_poly(source, resample_rate, audio_sr)
                    cgram_comparison_dict[sound_fn][H_key].append(cutil.make_gammatonegram(wav_rs, resample_rate))
                else:
                    cgram_comparison_dict[sound_fn][H_key].append(cutil.make_gammatonegram(source, audio_sr))

    return cgram_comparison_dict, compete_hypotheses


def compare(net_results, explanations, sound_fns, compete_hypotheses):
    """ Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")
    judgments = np.full((len(sound_fns), len(compete_hypotheses)), np.nan)
    for cond_idx, condition in enumerate(sound_fns):
        network_outputs = net_results[condition]
        comparison_gtgs = explanations[condition]

        distance_matrices = {}
        together_hypotheses_idxs = [0, 1]
        other_idxs = range(2, 10)

        # Get the distance between every triplet of standards, account for ordering
        # This provides the denominator for the preference function
        normalizing_distances = np.full((3,3,3,len(together_hypotheses_idxs),len(other_idxs)), np.nan)
        for t_idx in together_hypotheses_idxs:
            Ht = compete_hypotheses[t_idx]
            Hcgs = comparison_gtgs[Ht]
            for o_idx in other_idxs: 
                Ot = compete_hypotheses[o_idx]
                Ocgs = comparison_gtgs[Ot]
                mini_d_matrix = np.full((3, 3), np.nan)
                for n_idx_1 in range(3):
                    for n_idx_2 in range(3):
                        for n_idx_3 in range(3):
                            norm_gram_1 = Hcgs[n_idx_1]
                            norm_gram_2 = Hcgs[n_idx_2]
                            norm_gram_3 = Hcgs[n_idx_3]
                            if (n_idx_1 == n_idx_2) or (n_idx_1 == n_idx_3) or (n_idx_3 == n_idx_2):
                                normalizing_distances[n_idx_1, n_idx_2, n_idx_3, t_idx, o_idx-2] = np.inf
                            else:
                                d1=distance_metric(Ocgs[0], norm_gram_1, Ocgs[0]>20)
                                d2=distance_metric(Ocgs[1], norm_gram_2, Ocgs[1]>20)
                                d3=distance_metric(Ocgs[2], norm_gram_3, Ocgs[2]>20)
                                normalizing_distances[n_idx_1, n_idx_2, n_idx_3, t_idx, o_idx-2] = cutil.combine_distances([d1, d2, d3])

        # Find best distance for each hypothesis
        for k_idx, k in enumerate(compete_hypotheses):
            if k not in comparison_gtgs.keys():
                continue
            if len(network_outputs) < len(comparison_gtgs[k]):
                continue

            # find distance between neural networks and standard of this hypothesis
            compare_to_explanations = np.zeros((len(network_outputs), len(comparison_gtgs[k])))
            for v_idx, v in enumerate(comparison_gtgs[k]):
                mask = v > v.min()
                for n_idx, network_output in enumerate(network_outputs):
                    compare_to_explanations[n_idx, v_idx] = distance_metric(v, network_output, mask) 

                # Find best permutation of network outputs that minimize distance
                if len(comparison_gtgs[k]) == 2:
                    distance_matrices[k] =  np.zeros((len(network_outputs), len(network_outputs)))
                    for a_idx, a_distance in enumerate(compare_to_explanations[:, 0].tolist()):
                        for b_idx, b_distance in enumerate(compare_to_explanations[:, 1].tolist()):
                            if a_idx == b_idx:
                                distance_matrices[k][a_idx, b_idx] = np.inf
                            else:
                                distance_matrices[k][a_idx, b_idx] = cutil.combine_distances([a_distance, b_distance])
                elif len(comparison_gtgs[k]) == 3:
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

            # Normalizing distance should compare hypotheses in the other group
            if k_idx < 2:
                norm_dist = np.min(normalizing_distances[:, :, :, k_idx, :])
            else:
                norm_dist = np.min(normalizing_distances[:, :, :, :, k_idx - 2])
            judgments[cond_idx, k_idx] = np.min(distance_matrices[k])/norm_dist

    print(judgments)
    return judgments


def main(sound_group, network, expt_name=None):

    # Load in network results
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    net_results = {}
    frequencies = [
        [2800, 1556, 600, 333],
        [600, 333, 2800, 1556],
        [2800, 2642, 1556, 1468],
        [333, 314, 600, 566],
        [2800, 1556, 2642, 1468],
        [600, 333, 566, 314],
        [2800, 600, 1468, 314]
        ]
    conditions = ['isolate', 'isolate', 'isolate', 'isolate', 'absorb',' absorb', 'absorb']
    sound_fns = []
    for i in range(len(conditions)):
        sound_fns.append(f"{conditions[i]}_A{frequencies[i][0]}_B{frequencies[i][1]}_D1-{frequencies[i][2]}_D2-{frequencies[i][3]}")

    for sound_fn in sound_fns:
        fns = glob(netpath + f"{sound_fn}_*s*_estimate.wav")
        network_outputs = []
        for fn in fns:
            network_output, net_sr = sf.read(fn)
            network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
        # Make sure every network output is a triplet
        if len(network_outputs) == 1:
            network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
        if len(network_outputs) == 2:
            network_outputs.append(cutil.make_gammatonegram(np.zeros(network_output.shape), net_sr))
        net_results[sound_fn] = network_outputs

    # Get comparison stimuli and resample if necessary
    comparison_dicts, compete_hypotheses = generate_comparison_sounds(network, sound_fns, resample_rate=net_sr if network != "sequential-gen-model" else None)

    # Compare in gammatonegram space
    judgments = compare(net_results, comparison_dicts, sound_fns, compete_hypotheses)
    # Use psychophysics.analysis code to process
    model_results = plot_compete_average(
        -judgments[:, :, None], compete_hypotheses,
        sound_fns, None, sound_group, network,
        savepath=os.path.join(netpath, "result_compete.png")
        )
    print(os.path.join(netpath, "result_compete.png"))
    np.save(
        os.path.join(netpath, "result_compete.npy"),
        {"conditions": sound_fns, "d": model_results}
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound_group", type=str, help="which sounds do you want to analyze?")
    parser.add_argument("--expt_name", default=None, type=str)
    args = parser.parse_args()
    print("Plotting figure...")
    main(args.sound_group, args.network, expt_name=args.expt_name)