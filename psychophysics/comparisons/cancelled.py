import os
import numpy as np
from glob import glob
import soundfile as sf
import scipy.signal
from scipy.fft import rfft, rfftfreq

import psychophysics.analysis.cancelled as pac


def get_spectrum(s, audio_sr):
    n = len(s)
    yf = rfft(s)
    xf = rfftfreq(n, 1 / audio_sr)
    return xf, np.abs(yf)


def get_network_estimate(netpath, sound_name, sound_group, network, f0, harmonic_idx):
    """ Get the frequency of the most whistle-like output
        Whistle will have highest 1st to 2nd peak ratio
    """

    network_outputs = []
    for f in glob(netpath + f"{sound_name}_*_estimate.wav"):
        network_output, audio_sr = sf.read(f)
        network_outputs.append(network_output)

    if len(network_outputs) == 0:
        return None

    # Logic: the whistle stimulus is the one with much higher energy at one frequency than the rest
    peaks_ratios = []
    best_peaks = []
    for network_output in network_outputs:
        f, spectrum = get_spectrum(network_output, audio_sr)
        df = f[1] - f[0]
        f0ninety_idx = np.argmin(np.abs(f - 0.90*f0))
        spectrum[:f0ninety_idx] = 0
        peaks, _ = scipy.signal.find_peaks(spectrum, distance=int(0.95*f0/df))
        if len(peaks) == 0:
            continue
        peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)
        peaks_ratios.append(spectrum[peaks[0]]/spectrum[peaks[1]])
        best_peaks.append(f[peaks[0]])

    # Get closest guess of which is the whistle
    # If it's a whistle, the highest energy in the stimulus should be the fundamental frequency
    whistle_idx = np.argmax(peaks_ratios)

    return best_peaks[whistle_idx]


def fig(sound_group, network, expt_name=None, error_limit=100):

    netpath = os.path.join(os.environ["home_dir"], "comparisons", "results", network, sound_group, "")
    if network == "sequential-gen-model":
        netpath = os.path.join(netpath, expt_name, "")
    sound_path = os.path.join(os.environ["sound_dir"], sound_group, "")
    settings = np.load(os.path.join(sound_path, "classic_settings.npy"), allow_pickle=True).item()
    n_harmonics_to_test = settings["cancelled_harmonic_idxs"]
    sound_names = glob(sound_path + "*.wav")
    sound_names = [os.path.splitext(os.path.basename(sn))[0] for sn in sound_names]
    f0s = np.unique([int(sound_name.split("_")[1][1:]) for sound_name in sound_names])

    results_per_f0 = {}
    results_per_idx = {}
    for f0 in f0s:
        results_per_f0[f0] = {
            "harmonic_idxs": [],
            "percent_errors": [],
            "whistle_present": []
            }
        sound_names_f0 = [sn for sn in sound_names if "_f{:04d}_".format(f0) in sn]
        for sound_name in sound_names_f0:
            _, harmonic_idx = pac.get_f0_and_harmonic_idx(sound_name)
            f0_estimate = get_network_estimate(netpath, sound_name, sound_group, network, f0, harmonic_idx)
            # Translate the f0_estimate into %error for judgment
            if f0_estimate is not None:
                _, error = pac.compute_error_in_percent(sound_name, f0_estimate)
            if (error*100 > error_limit) or (f0_estimate is None):
                f0_estimate = None
            if f0_estimate is None:  # no whistle source was found
                harmonic_idx, error = pac.compute_error_in_percent(sound_name, 0)
                results_per_f0[f0]["harmonic_idxs"].append(harmonic_idx)
                percent_error = error_limit
                whistle_present = False
            else:
                harmonic_idx, error = pac.compute_error_in_percent(sound_name, f0_estimate)
                results_per_f0[f0]["harmonic_idxs"].append(harmonic_idx)
                percent_error = 100*error
                whistle_present = True
            # Format results
            results_per_f0[f0]["percent_errors"].append(percent_error)
            results_per_f0[f0]["whistle_present"].append(whistle_present)
            if harmonic_idx in results_per_idx.keys():
                results_per_idx[harmonic_idx]["f0s"].append(f0)
                results_per_idx[harmonic_idx]["percent_errors"].append(percent_error)
                results_per_idx[harmonic_idx]["whistle_present"].append(whistle_present)
            else:
                results_per_idx[harmonic_idx] = {
                    "f0s": [f0],
                    "percent_errors": [percent_error],
                    "whistle_present": [whistle_present]
                    }

    pac.plotter(
        f0s, results_per_f0, n_harmonics_to_test,
        netpath, network, sound_group,
        ylimit=error_limit, text_placement=error_limit-30
        )
    trial_proportions = pac.plot_trial_proportion(
        n_harmonics_to_test, results_per_idx, network, sound_group, netpath
        )
    model_results = {
        "harmonic_idxs": n_harmonics_to_test,
        "p_beyond_2": trial_proportions
        }
    np.save(os.path.join(netpath, "result.npy"), model_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--sound-group", type=str, help="which sounds do you want to analyze?")
    parser.add_argument("--expt-name", default=None, type=str, help="which inferences do you want to analyze?")
    args = parser.parse_args()
    print("Plotting figure...")
    fig(args.sound_group, args.network, expt_name=args.expt_name)
