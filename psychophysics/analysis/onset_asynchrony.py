import os
import json
from glob import glob
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.interpolate
import scipy.stats
from scipy.special import logsumexp

import matplotlib.pyplot as plt
import torch

import renderer.util
import inference.io
import inference.metrics
import inference.samplers
import psychophysics.analysis.thresholds as pat


def categorize_hillenbrand():
    """Get distributions of vowel formants based on data at
        https://homepages.wmich.edu/~hillenbr/voweldata/vowdata.dat
        col0:  filename (character1, m=man; chars2-3: speaker_id, chars4-5: eh="head", ih="hid")
        col2:  f0 at "steady state"
        col3:  F1 at "steady state"
        col4:  F2 at "steady state"
    """

    raw_data = np.loadtxt(
        os.path.join(os.environ["home_dir"], "psychophysics", "analysis", "cache", "vowels.txt"),
        dtype=str
        )
    T = pd.DataFrame(raw_data[1:, :])
    speaker_mu_and_std = speaker_normalization(T)

    ih_f1 = []
    ih_f2 = []
    eh_f1 = []
    eh_f2 = []
    for i in range(T.shape[0]):
        if "eh" in T[0][i]:
            eh_f1.extend(zscore(T, i, 1, speaker_mu_and_std))
            eh_f2.extend(zscore(T, i, 2, speaker_mu_and_std))
        elif "ih" in T[0][i]:
            ih_f1.extend(zscore(T, i, 1, speaker_mu_and_std))
            ih_f2.extend(zscore(T, i, 2, speaker_mu_and_std))

    ih_data = np.concatenate((
            np.array(ih_f1)[:, None], np.array(ih_f2)[:, None]
        ), axis=1)
    eh_data = np.concatenate((
        np.array(eh_f1)[:, None], np.array(eh_f2)[:, None]
        ), axis=1)
    ih_distribution = (np.mean(ih_data, axis=0), np.cov(ih_data.T))
    eh_distribution = (np.mean(eh_data, axis=0), np.cov(eh_data.T))

    return ih_distribution, eh_distribution, ih_data, eh_data


def zscore(T, row_idx, formant_idx, all_speaker_formant_vals):
    col_idx = 3 if formant_idx == 1 else 4
    m = np.nanmean(all_speaker_formant_vals[formant_idx])
    s = np.nanstd(all_speaker_formant_vals[formant_idx])
    return [(float(T[col_idx][row_idx]) - m) / s]


def speaker_normalization(T):
    """ Adank et al., 2004: A comparison of vowel normalization procedures
        for language variation research
    See https://asa.scitation.org/doi/pdf/10.1121/1.1795335?casa_token=OTsx2_a4fRwAAAAA:pi5wR4CcxmLEsHqwT-mnqLnWsqkFT5EaIaYFYhkQ9mc8ru_CStx50NnAI-VIoHnzUWIIxDssFezv
    """
    speaker_mu_and_std = {1: [], 2: []}
    for speaker_idx in range(1, 51):
        speaker_formant_vals = {1: [], 2: []}
        for i in range(T.shape[0]):
            this_speaker = ("m" == T[0][i][0]) and (f"{speaker_idx:02d}" == T[0][i][1:3])
            if this_speaker and ("eh" in T[0][i] or "ih" in T[0][i]):
                speaker_formant_vals[1].append(int(T[3][i]))
                speaker_formant_vals[2].append(int(T[4][i]))
        for formant_idx in [1, 2]:
            speaker_mu_and_std[formant_idx].append((
                np.mean(speaker_formant_vals[formant_idx]),
                np.std(speaker_formant_vals[formant_idx])
            ))
    return speaker_mu_and_std


def classify(conditions, formants_by_condition, settings, sound_group, expt_name, savepath="./", model_comparison_dir=None):

    print("Classifying formants as vowels!")
    # Get means and standard deviations for hillenbrand and expt data
    (ih_mean, ih_cov), (eh_mean, eh_cov), ih_data, eh_data = categorize_hillenbrand()
    n_seeds = len(formants_by_condition[(conditions[0], settings["F1s"][0])][0])
    all_f1s = []
    all_f2s = []
    for condition in conditions:
        for F1 in settings["F1s"]:
            for seed in range(n_seeds):
                f1 = formants_by_condition[(condition, F1)][0][seed]
                f2 = formants_by_condition[(condition, F1)][1][seed]
                all_f1s.append(f1)
                all_f2s.append(f2)

    # Get z-score of experiment formants
    category_proportions = {}
    for condition in conditions:
        category_proportions[condition] = []
        for F1 in settings["F1s"]:
            proportion_in_condition = []
            for seed in range(n_seeds):
                f1 = formants_by_condition[(condition, F1)][0][seed]
                f2 = formants_by_condition[(condition, F1)][1][seed]
                normalized_f1 = (f1 - np.mean(all_f1s)) / np.std(all_f1s)
                normalized_f2 = (f2 - np.mean(all_f2s)) / np.std(all_f2s)
                pih = scipy.stats.multivariate_normal.pdf(
                    np.array([normalized_f1, normalized_f2]),
                    mean=ih_mean, cov=ih_cov
                    )
                peh = scipy.stats.multivariate_normal.pdf(
                    np.array([normalized_f1, normalized_f2]),
                    mean=eh_mean, cov=eh_cov
                    )
                proportion_in_condition.append(pih / (pih + peh))
            category_proportions[condition].append(proportion_in_condition)

    # Calculate a boundary for each seed and then take a mean over seeds
    result = {}
    stderrs = {}
    for condition in conditions:
        print(condition)
        boundaries = []
        for seed in range(n_seeds):
            y = np.array([f[seed] for f in category_proportions[condition]])
            boundary = pat.average_crossing_points(settings["F1s"], y)
            boundaries.append(boundary)
        print(f"{condition}: ", boundaries)
        result[condition] = np.mean(boundaries)
        stderrs[condition] = np.std(boundaries)/np.sqrt(len(boundaries))
    print(result)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5.2, 2.2))
    # Human results
    x = [0, 1, 2, 3]
    axs[0].bar(
        x, [461, 422.5, 454, 457.5],
        color="grey", tick_label=["Basic", "Shifted", "32ms", "240ms"]
        )
    axs[0].errorbar(
        x, [461, 422.5, 454, 457.5], [0, 4.5, 1.25, 3],
        fmt=" ", capsize=2, ecolor="black"
        )
    axs[0].set_ylim([461-50, 461+20])
    axs[0].set_title("Darwin and Sutherland (1984)")
    # Modle results
    condition_order = ["basic", "on0_off0", "on32_off0", "on240_off0"]
    axs[1].bar(
        x,
        [result[s] for s in condition_order],
        color="grey"
        )
    axs[1].errorbar(
        x,
        [result[s] for s in condition_order],
        yerr=[stderrs[s] for s in condition_order],
        fmt=" ", capsize=2, ecolor="black"
        )
    axs[0].set_ylabel("Vowel threshold\nNominal F1 (Hz)")
    plt.tight_layout()
    print(os.path.join(savepath, "oa_expt1_boundary.png"))
    plt.savefig(os.path.join(savepath, "oa_expt1_boundary.png"))
    plt.savefig(os.path.join(savepath, "oa_expt1_boundary.svg"))
    plt.close()

    if model_comparison_dir is not None:
        comparison_dict = {}
        comparison_dict["condition"] = ['basic', 'on0_off0', 'on32_off0', 'on240_off0']
        comparison_dict["boundary"] = {k: result[k] for k in comparison_dict["condition"]}
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result.npy"),
            comparison_dict, allow_pickle=True
            )
        for_plots = {"stderrs": stderrs, "means": result}
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots.npy"),
            for_plots, allow_pickle=True
            )

    return


def define_latents_logger(vowel_idx):
    """ Get latent variables from inference scene object """
    def latents_logger(scene):
        # Get timing information
        trimmer = scene.sources[vowel_idx].renderer.trimmer
        window_ts = trimmer.ty_trim
        onset = scene.sources[vowel_idx].sequence.events[0].onset.timepoint
        offset = scene.sources[vowel_idx].sequence.events[0].offset.timepoint
        # Get amplitude/spectrum information
        f0 = scene.sources[vowel_idx].renderer.AM_and_filter.f0_for_shift
        tf_grid = scene.sources[vowel_idx].renderer.AM_and_filter.tf_grid.permute(0,2,1)
        shifted_grid = scene.sources[vowel_idx].renderer.AM_and_filter.shift_E(tf_grid, f0, V=0.0)
        inactive = (shifted_grid == 0.0)
        mean_spectrum_over_time = torch.zeros(shifted_grid.shape[0], shifted_grid.shape[1])
        # Get the active timepoints that occur within the event boundaries
        for batch_idx in range(onset.shape[0]):
            this_window_ts = window_ts[batch_idx, :]
            active_timepoints = (onset[batch_idx] <= this_window_ts)*(this_window_ts <= offset[batch_idx])
            active_spectrum = shifted_grid[batch_idx, :, active_timepoints]
            inactive_to_use = inactive[batch_idx, :, active_timepoints]
            mean_spectrum_over_time[batch_idx, :] = active_spectrum.sum(1)/(1e-12 + (1.0*(~inactive_to_use))).sum(1)
        squeezed_x = scene.sources[vowel_idx].gps.spectrum.feature.gp_x.squeeze().detach().cpu().numpy()
        if len(squeezed_x.shape) > 1:
            squeezed_x = squeezed_x[0, :]
        spectrum = {
            "x": squeezed_x,
            "y": mean_spectrum_over_time.detach().cpu().numpy()
            }
        return spectrum
    return latents_logger


def define_latents_logger_no_trimmer(vowel_idx):
    """ Get latents from inference scene object, no trimmer used """
    def latents_logger(scene):
        # Get timing information
        window_ts = scene.sources[vowel_idx].gps["f0"].feature.gp_x[:, :, 0]
        onset = scene.sources[vowel_idx].sequence.events[0].onset.timepoint
        offset = scene.sources[vowel_idx].sequence.events[0].offset.timepoint
        # Get amplitude/spectrum information
        f0 = scene.sources[vowel_idx].renderer.AM_and_filter.f0_for_shift
        tf_grid = scene.sources[vowel_idx].renderer.AM_and_filter.tf_grid.permute(0,2,1)
        shifted_grid = scene.sources[vowel_idx].renderer.AM_and_filter.shift_E(tf_grid, f0, V=0.0)
        inactive = (shifted_grid == 0.0)
        mean_spectrum_over_time = torch.zeros(shifted_grid.shape[0], shifted_grid.shape[1])
        # Get the active timepoints that occur within the event boundaries
        for batch_idx in range(onset.shape[0]):
            this_window_ts = window_ts[batch_idx, :]
            active_timepoints = (onset[batch_idx] <= this_window_ts)*(this_window_ts <= offset[batch_idx])
            active_spectrum = shifted_grid[:, :, 1:-1][batch_idx, :, active_timepoints]
            inactive_to_use = inactive[:, :, 1:-1][batch_idx, :, active_timepoints]
            mean_spectrum_over_time[batch_idx, :] = active_spectrum.sum(1)/(1e-12 + (1.0*(~inactive_to_use))).sum(1)
        squeezed_x = scene.sources[vowel_idx].gps.spectrum.feature.gp_x.squeeze().detach().cpu().numpy()
        if len(squeezed_x.shape) > 1:
            squeezed_x = squeezed_x[0, :]
        spectrum = {
            "x": squeezed_x,
            "y": mean_spectrum_over_time.detach().cpu().numpy()
            }
        return spectrum
    return latents_logger


def get_explanation_from_inits(inference_dir, sound_name, sound_group, expt_name):
    """ 1. Find the highest performing samples for each stimulus in the experiment """
    expt_folders = glob(
        os.path.join(inference_dir, expt_name, sound_group, sound_name, "*/")
        )
    initializations = []
    for initialization in expt_folders:
        initializations.append(initialization.split(os.sep)[-2].split("-seed")[0])
    initializations = np.unique(initializations)
    print("Initializations for {}: ".format(sound_name), initializations)
    results_dict = {}

    for init_group in initializations:
        seed_initializations = glob(
            os.path.join(inference_dir, expt_name, sound_group, sound_name, f"*{init_group}-*/")
            )
        print(f"n_seeds for {initialization}:", len(seed_initializations))

        for seed_idx, initialization in enumerate(seed_initializations):
            scene, ckpt = inference.io.restore_hypothesis(initialization)
            config = inference.io.get_config(initialization)
            if "trim" in config["renderer"]["source"]["harmonic"].keys():
                if config["renderer"]["source"]["harmonic"]["trim"] is False:
                    latent_logger = define_latents_logger_no_trimmer(0)
                else:
                    latent_logger = define_latents_logger(0)
            else:
                latent_logger = define_latents_logger(0)

            scene = ckpt["metrics"].best_scene
            spectrum = latent_logger(scene)
            spectrum["y"] = spectrum["y"].mean(0)

            results_dict[(init_group, seed_idx)] = {
                "elbo": ckpt["metrics"].elbo(),
                "spectrum": spectrum
            }

    return results_dict


def get_sound_from_iterative(inference_dir, sound_name, sound_group, expt_name):
    """ Get sound from sequential inference """

    sound_folder = os.path.join(
        inference_dir, expt_name, sound_group, sound_name, ""
        )
    with open(sound_folder + "results.json", "r") as f:
        results = json.load(f)
    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons", "results", "gen-model",
        sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)

    round_idx = len(results["keep"])-1
    expl_idx = int(results["keep"][-1][0])

    this_initialization = os.path.join(
        inference_dir, expt_name,
        sound_group, sound_name, f"round{round_idx:03d}-{expl_idx:03d}", ""
        )
    _, ckpt = inference.io.restore_hypothesis(this_initialization)
    scene = ckpt["metrics"].best_scene
    batch_idx = ckpt["metrics"].best_scene_scores.argmax()
    for source_idx, source in enumerate(scene.sources):
        sf.write(
            os.path.join(
                sequential_folder,
                f"{sound_name}_round{round_idx:03d}-{expl_idx:03d}_source{source_idx:02d}_estimate.wav"
                ),
            source.source_wave.detach().numpy()[batch_idx, :], scene.audio_sr
            )

    return None


def get_formant_in_range(spectrum, start_Hz, end_Hz):
    """ Start stop and step in ERB """
    (start, stop, step) = (renderer.util.freq_to_ERB(start_Hz), renderer.util.freq_to_ERB(end_Hz), 0.1)
    x = spectrum["x"]
    y = spectrum["y"]
    interpol8r = scipy.interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
    finely_sampled_xs = np.arange(start, stop, step)
    spectrum_in_range = interpol8r(finely_sampled_xs)
    return finely_sampled_xs[np.argmax(spectrum_in_range)]


def get_formant_from_multiple(results_dict):
    """ Start stop and step in ERB """

    inits = np.unique([
        init for (init, seed_idx) in results_dict.keys()
        ])
    seeds = np.unique([
        seed_idx for (init, seed_idx) in results_dict.keys()
        ])

    elbo = np.array([
        [results_dict[(init, seed)]['elbo'] for seed in seeds]
        for init in inits
    ])
    spectrum = np.stack([
        np.stack([results_dict[(init, seed)]['spectrum']["y"] for seed in seeds])
        for init in inits
    ])  # Shape: len(inits), len(seeds), len(spectrum)

    if "vowel" in inits and "vowel+tone" in inits:  # basic and shifted continua
        first_formant = np.array([
            [
                get_formant_in_range(
                    results_dict[(init, seed)]['spectrum'],
                    start_Hz=325, end_Hz=550
                ) for seed in seeds
            ] for init in inits
        ])

        second_formant = np.array([
            [
                get_formant_in_range(
                    results_dict[(init, seed)]['spectrum'],
                    start_Hz=2000, end_Hz=2500
                ) for seed in seeds
            ] for init in inits
        ])

        init_to_use = elbo.argmax(0)
        elbo = elbo.max(0)
        first_formant = renderer.util.ERB_to_freq(np.array([
            first_formant[init_to_use[i], i] for i in range(first_formant.shape[1])
        ]))
        second_formant = renderer.util.ERB_to_freq(np.array([
            second_formant[init_to_use[i], i] for i in range(second_formant.shape[1])
        ]))

        return (first_formant, second_formant, elbo)

    elif "basic-init" in inits and "onzero-init" in inits:  # Asynchronous continua
        
        weights = np.exp(elbo - logsumexp(elbo, axis=0, keepdims=True))

        first_formant = renderer.util.ERB_to_freq(np.array([[
            get_formant_in_range({
                    "x": results_dict[(inits[0], seed)]['spectrum']["x"],
                    "y": spectrum[init_idx, seed_idx, :]
                }, start_Hz=325, end_Hz=550)
            for seed_idx, seed in enumerate(seeds)
        ] for init_idx, init in enumerate(inits)]))

        second_formant = renderer.util.ERB_to_freq(np.array([[
            get_formant_in_range({
                    "x": results_dict[(inits[0], seed)]['spectrum']["x"],
                    "y": spectrum[init_idx, seed_idx, :]
                }, start_Hz=2000, end_Hz=2500)
            for seed_idx, seed in enumerate(seeds)
        ] for init_idx, init in enumerate(inits)]))

        weighted_first_formant = np.sum(weights * first_formant, axis=0)
        weighted_second_formant = np.sum(weights * second_formant, axis=0)

        return (weighted_first_formant, weighted_second_formant, elbo)


def expt1(sound_group, expt_name, inference_dir, sound_dir, savepath, model_comparison_dir=None):

    settings_path = os.path.join(sound_dir, sound_group, "")
    settings = np.load(
        os.path.join(settings_path, "oa_expt1_settings.npy"),
        allow_pickle=True
    ).item()
    stimulus_types = ["basic", "on0_off0", "on32_off0", "on240_off0"]

    results_fn = os.path.join(savepath, "oa_expt1_results.npy")
    results_spectra = {}
    for stimulus_type in stimulus_types:
        for F1 in settings["F1s"]:
            if stimulus_type not in results_spectra.keys():
                results_spectra[stimulus_type] = {}
            if F1 not in results_spectra[stimulus_type].keys():
                sound_name = f"{F1}_{stimulus_type}"
                print(f"Sampling for {sound_name}", flush=True)
                spectrum = get_explanation_from_inits(
                    inference_dir, sound_name, sound_group, expt_name
                    )
                results_spectra[stimulus_type][F1] = spectrum
                np.save(results_fn, results_spectra)

    formants_by_condition = {}
    for stimulus_type in stimulus_types:
        for F1 in settings["F1s"]:
            formants_by_condition[(stimulus_type, F1)] = get_formant_from_multiple(results_spectra[stimulus_type][F1])
    classify(stimulus_types, formants_by_condition, settings, sound_group, expt_name, savepath=savepath, model_comparison_dir=model_comparison_dir)

    return results_spectra


if __name__ == '__main__':
    print("Analyzing onset asynchrony!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str, help="folder to save wavs")
    parser.add_argument("--expt-name", type=str, help="folder to group inference results together")
    parser.add_argument("--inference-dir", type=str, default=os.environ["inference_dir"], help="top-level folder for inference")
    parser.add_argument("--sound-dir", type=str, default=os.environ["sound_dir"], help="top-level folder for inference")
    parser.add_argument("--results-dir", type=str, default="home")
    parser.add_argument("--model-comparison-dir", type=str, help="where to save results for model comparisons", required=False)
    args = parser.parse_args()
    if args.results_dir == "home":
        results_dir = os.path.join(args.inference_dir, args.expt_name, args.sound_group, "")
    else:
        results_dir = args.results_dir
    expt1(args.sound_group, args.expt_name, args.inference_dir, args.sound_dir, results_dir,  model_comparison_dir=args.model_comparison_dir)
