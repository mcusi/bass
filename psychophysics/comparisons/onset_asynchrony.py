import numpy as np
import scipy.signal
import os
from glob import glob
import soundfile as sf
from copy import deepcopy

import math
from scipy.signal import lfilter
from librosa import lpc

from util import context
import psychophysics.comparisons.cutil as cutil
import psychophysics.generation.onset_asynchrony as gen
import psychophysics.analysis.onset_asynchrony as oa_analyze


def generate_comparison_sounds(settings, network, resample_rate=None):
    """ Comparison sounds to convert network soundwaves into judgments
        Here we use the vowels, so we can find the vowels in the network
        output and then measure its formants.
    """

    audio_sr = 20000
    context(audio_sr=audio_sr, rms_ref=1e-6)
    # Use an extended set of formants to make sure we find the vowel
    comparison_F1s = settings["F1s"]
    overwrite = deepcopy(settings)
    overwrite["F1s"] = comparison_F1s
    ds_dict, settings = gen.expt1("", overwrite=overwrite, save_wav=False)
    audio_sr = settings["audio_sr"]

    for k, sound in ds_dict.items():
        if network == "sequential-gen-model":
            # Account for pad from amortized neural network duration requirement
            seconds_per_frame = 0.005
            if len(ds_dict[k]) % int(np.round(seconds_per_frame*audio_sr)) != 0:
                N = int(np.round(seconds_per_frame*audio_sr))
                diff = int(np.ceil(len(ds_dict[k]) / N) * N) - len(ds_dict[k])
                ds_dict[k] = np.concatenate((ds_dict[k], np.zeros((diff,))))
        if resample_rate is not None:
            wav_rs = scipy.signal.resample_poly(sound, resample_rate, audio_sr)
            ds_dict[k] = cutil.make_gammatonegram(wav_rs, resample_rate)
        else:
            ds_dict[k] = cutil.make_gammatonegram(ds_dict[k], audio_sr)

    return ds_dict, comparison_F1s


def find_vowels(net_results, fig1_settings, conditions, sounds_dict, comparison_F1s, net_sr):
    """ Select which network output is the vowel by comparing to standards """

    distance_metric = cutil.get_distance_metric("normal_l2_distance")
    formants_by_condition = {}
    for condition in conditions:
        for F1_idx, F1 in enumerate(fig1_settings["F1s"]):
            sound_name = "_".join([str(F1), condition])
            network_outputs = net_results["grams"][sound_name]
            network_sounds = net_results["sounds"][sound_name]

            # Compute the distance between every standard and every network output
            distance_matrix = np.zeros((len(comparison_F1s)*2, len(network_outputs)))
            comparison_idx = 0
            for compare_condition in ["basic", "on0_off0"]:
                for compare_F1 in comparison_F1s:
                    key = "_".join([str(compare_F1), compare_condition])
                    mask = sounds_dict[key] > sounds_dict[key].min()
                    for network_idx, network_output in enumerate(network_outputs):
                        distance_matrix[comparison_idx, network_idx] = distance_metric(sounds_dict[key], network_output, mask)
                    comparison_idx += 1

            # Choose the network output which minimizes the distance to any standard
            network_vowel_idx = np.argmin(np.min(distance_matrix, axis=0))
            # Measure the formants from the network output
            f1, f2 = get_formants_from_audio(network_sounds[network_vowel_idx], net_sr)
            formants_by_condition[(condition, F1)] = (f1, f2)

    return formants_by_condition


def get_formants_from_audio(x, Fs):
    """ Use linear predictve coding to get formants from vowel
        Used as references:
            https://github.com/manishmalik/Voice-Classification/blob/master/rootavish/formant.py
            https://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
            https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
    """

    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Pre-emphasis: apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    
    # Get LPC.
    # MATLAB tutorial gives notes on specifying model order
    # MATLAB tutorial:  To specify the model order, use the general rule that the order is two times the expected number of formants plus 2. In the frequency range, [0,|Fs|/2], you expect three formants
    # We know we gave Pratt 5 formants
    ncoeff = int(2 + 5)
    A = lpc(x1, ncoeff)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    unsorted_freqs = angz * (Fs / (2 * math.pi))
    sort_idxs = np.argsort(unsorted_freqs)
    frqs = unsorted_freqs[sort_idxs]
    bw = -1/2*(Fs/(2*np.pi))*np.log(np.abs([rts[s] for s in sort_idxs]));

    # MATLAB: rejects frequencies based on freq and bw
    # We set the bandwidth of the first formant to be 70Hz, and the fundamental is 125
    frqs = [frqs[i] for i in range(len(frqs)) if frqs[i] > 125]
    # here we use 325<f<550 for the first formant and 2000<f<2500 for the second
    first_formants = [f for f in frqs if 325 < f and f < 550]
    if len(first_formants) == 0:
        # Choose the formant closest to this range
        print("No first formants found!")
        bottom_closest = np.abs(np.asarray(frqs) - 325)
        top_closest = np.abs(np.asarray(frqs) - 550)
        first_formants = [frqs[np.argmin(bottom_closest)] if np.min(bottom_closest) < np.min(top_closest) else frqs[np.argmin(top_closest)]]
    second_formants = [f for f in frqs if 2000 < f and f < 2500]
    if len(second_formants) == 0:
        # Choose the formant closest to this range
        print("No second formants found!")
        bottom_closest = np.abs(np.asarray(frqs) - 2000)
        top_closest = np.abs(np.asarray(frqs) - 2500)
        second_formants = [frqs[np.argmin(bottom_closest)] if np.min(bottom_closest) < np.min(top_closest) else frqs[np.argmin(top_closest)]]

    return np.mean(first_formants) if len(first_formants) > 0 else np.nan, np.mean(second_formants) if len(second_formants) > 0 else np.nan


def fig1(sound_group, network, expt_name=None):

    # Load stimulus generation settings
    soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
    fig1_settings = np.load(soundpath + "oa_expt1_settings.npy", allow_pickle=True).item()
    F1s = fig1_settings["F1s"]
    conditions = ["basic", "on0_off0", "on32_off0", "on240_off0"]

    # Get network results
    print("Getting network results...")
    net_results = {"grams": {}, "sounds": {}}
    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    for F1 in F1s:
        for condition in conditions:
            sound_name = "_".join([str(F1), condition])
            network_outputs = []
            network_sounds = []
            for f in glob(os.path.join(netpath, f"{sound_name}_*_estimate.wav")):
                network_output, net_sr = sf.read(f)
                network_sounds.append(network_output)
                network_outputs.append(cutil.make_gammatonegram(network_output, net_sr))
            net_results["grams"][sound_name] = network_outputs
            net_results["sounds"][sound_name] = network_sounds

    # Get comparison sounds
    sounds_dict, comparison_F1s = generate_comparison_sounds(
        fig1_settings, network, resample_rate=None if network == "sequential-gen-model" else net_sr
        )
    # Select network output and easure formants
    formants_by_condition = find_vowels(
        net_results, fig1_settings, conditions,
        sounds_dict, comparison_F1s, net_sr
        )
    savepath = os.path.join(netpath, "result.png")
    print(savepath)
    oa_analyze.classify(
        conditions, formants_by_condition,
        fig1_settings, "", "",
        savepath=netpath, model_comparison_dir=netpath
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound-group", type=str,
                        help="which sounds do you want to analyze?")
    parser.add_argument("--expt-name", default=None,
                        type=str,
                        help="which inferences do you want to analyze?")
    args = parser.parse_args()
    fig1(args.sound_group, args.network, expt_name=args.expt_name)
