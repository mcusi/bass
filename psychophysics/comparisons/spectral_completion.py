import os
import numpy as np
import scipy.signal
from glob import glob
import soundfile as sf

import psychophysics.hypotheses.spectral_completion_match as hyp
import psychophysics.analysis.spectral_completion as pasc
import psychophysics.comparisons.cutil as cutil
from util import context


def generate_comparison_sounds(figure_idx, settings, network, resample_rate=None):
    """Comparison sounds to convert network soundwaves into judgments"""

    context(audio_sr=20000, rms_ref=1e-6)
    if figure_idx == 1:

        middle_SLs = np.arange(-15, 40, 2.5)
        tab_dict, _, audio_sr = hyp.fig1_tabs(middle_SLs, settings)

        for k, sound in tab_dict.items():
            if "sequential-gen-model" in network:
                # Account for pad from amortized neural network duration requirement
                seconds_per_frame = 0.005
                if len(tab_dict[k]) % int(np.round(seconds_per_frame*audio_sr)) != 0:
                    N = int(np.round(seconds_per_frame*audio_sr))
                    diff = int(np.ceil(len(tab_dict[k]) / N) * N) - len(tab_dict[k])
                    tab_dict[k] = np.concatenate((tab_dict[k], np.zeros((diff,))))
            if resample_rate is not None:
                wav_rs = scipy.signal.resample_poly(sound, resample_rate, audio_sr)
                tab_dict[k] = cutil.make_gammatonegram(wav_rs, resample_rate)
            else:
                tab_dict[k] = cutil.make_gammatonegram(tab_dict[k], audio_sr)

    elif figure_idx == 2:

        middle_SLs = np.arange(-5, 13, 1.25)  # goes up to 12.5
        tab_dict, _, audio_sr = hyp.fig2_tabs(middle_SLs, settings)
        for stimulus_name, stimulus_dict in tab_dict.items():
            for middle_level, sound in stimulus_dict.items():
                if "sequential-gen-model" in network:
                    seconds_per_frame = 0.005
                    if len(tab_dict[stimulus_name][middle_level]) % int(np.round(seconds_per_frame*audio_sr)) != 0:
                        N = int(np.round(seconds_per_frame*audio_sr))
                        diff = int(np.ceil(len(tab_dict[stimulus_name][middle_level]) / N) * N) - len(tab_dict[stimulus_name][middle_level])
                        tab_dict[stimulus_name][middle_level] = np.concatenate((tab_dict[stimulus_name][middle_level], np.zeros((diff,))))
                if resample_rate is not None:
                    wav_rs = scipy.signal.resample_poly(sound, resample_rate, audio_sr)
                    tab_dict[stimulus_name][middle_level] = cutil.make_gammatonegram(wav_rs, resample_rate)
                else:
                    tab_dict[stimulus_name][middle_level] = cutil.make_gammatonegram(tab_dict[stimulus_name][middle_level],audio_sr)

    return tab_dict


def fig1_compare(net_results, sound_names, tab_dict):
    """Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")

    judgments = []
    for sound_name in sound_names:

        # Get distance to tabs
        network_outputs = net_results[sound_name]
        tab_distances = {}
        for k, v in tab_dict.items():
            tab_distances[k] = []
            mask = v > v.min()
            for network_output in network_outputs:
                tab_distances[k].append(distance_metric(v, network_output, mask))

        min_distances = sorted(
            tab_distances.keys(), key=lambda k: np.min(tab_distances[k])
            )
        judgments.append(min_distances[0])

    return judgments


def figure1(sound_group, network, expt_name=None):

    # Figure 1 sounds
    sound_names = ["i", "ii", "iii", "iv", "v"]

    # Load stimulus generation settings
    soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
    fig1_settings = np.load(
        soundpath + "sc_fig1_settings_seed0.npy", allow_pickle=True
        ).item()

    # Get network results
    print("Getting network results...")
    net_results = {}
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, args.expt_name, "")
    for sound_name in sound_names:
        network_outputs = []
        for f in glob(netpath + f"sc_1{sound_name}_*_estimate.wav"):
            network_output, net_sr = sf.read(f)
            network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
        net_results[sound_name] = network_outputs

    tab_dict = generate_comparison_sounds(
        1, fig1_settings, network,
        resample_rate=None if "sequential-gen-model" in network else net_sr
        )
    judgments = fig1_compare(net_results, sound_names, tab_dict)
    print(judgments)
    print(os.path.join(netpath, "sc_fig1.png"))
    pasc.fig1_plotter(
        sound_names, judgments, np.zeros(len(judgments)),
        os.path.join(netpath, "sc_fig1.png"), sound_group, network
        )
    model_results = {"fig1": judgments}
    return model_results


def fig2_compare(net_results, sound_names, tab_dict):
    """ Compare the "hypotheses" generated in `generate_comparison_sounds`
        with the neural outputs by taking the distance in spectrogram space
    """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")

    judgments = []
    for sound_name in sound_names:

        # Get distance to tabs
        network_outputs = net_results[sound_name]
        tab_distances = {}
        for k, v in tab_dict[sound_name].items():
            tab_distances[k] = []
            mask = v > v.min()
            for network_output in network_outputs:
                tab_distances[k].append(distance_metric(v, network_output, mask))

        min_distances = sorted(
            tab_distances.keys(), key=lambda k: np.min(tab_distances[k])
            )
        judgments.append(min_distances[0])

    return judgments


def figure2(sound_group, network, fig1_model_results=None, expt_name=None):

    # Figure 2 sounds
    sound_names = ["i", "ii", "iii", "iv", "v", "vi"]

    # Load stimulus generation settings
    soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
    fig2_settings = np.load(soundpath + "sc_fig2_settings_seed0.npy", allow_pickle=True).item()

    # Get network results
    print("Getting network results...")
    net_results = {}
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    for sound_name in sound_names:
        network_outputs = []
        for f in glob(netpath + f"sc_2{sound_name}_*_estimate.wav"):
            network_output, net_sr = sf.read(f)
            network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
        net_results[sound_name] = network_outputs

    # Get comparison stimuli results
    tab_dict = generate_comparison_sounds(
        2, fig2_settings, network,
        resample_rate=None if "sequential-gen-model" in network else net_sr
        )
    judgments = fig2_compare(net_results, sound_names, tab_dict)
    print(judgments)
    print(os.path.join(netpath, "sc_fig2.png"))
    pasc.fig2_plotter(
        sound_names, judgments, np.zeros(len(judgments)),
        os.path.join(netpath, "sc_fig2.png"), sound_group, network)
    if fig1_model_results is not None:
        fig1_model_results["fig2"] = judgments
        np.save(netpath + "result.npy", fig1_model_results)
    else:
        model_results = {"fig2": judgments}
        np.save(netpath + "result.npy", model_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound_group", type=str, help="which sounds do you want to analyze?")
    parser.add_argument("--expt_name", default=None, type=str, help="which sounds do you want to analyze?")
    args = parser.parse_args()
    print("Analyzing figure 1...")
    model_results = figure1(args.sound_group, args.network, expt_name=args.expt_name)
    print("Analyzing figure 2...")
    figure2(args.sound_group, args.network, fig1_model_results=model_results, expt_name=args.expt_name)