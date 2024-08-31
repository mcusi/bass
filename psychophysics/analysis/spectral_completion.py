import os
import json
import torch
import numpy as np
import scipy
from glob import glob
import matplotlib.pyplot as plt
import soundfile as sf

import inference.io

# Plotting colors
ratna = np.array([252., 186., 3.])/256.
vajra = np.array([74., 3., 252.])/256.


def define_latents_logger(tab_idx):
    """ Get latent variables out of the saved scenes """
    def latents_logger(scene):
        # Get timing information
        trimmer = scene.sources[tab_idx].renderer.trimmer
        window_ts = trimmer.t_y
        window_idxs = trimmer.window_idxs
        if scene.sources[tab_idx].sequence.n_events > 1:
            n_events = scene.sources[tab_idx].sequence.n_events
            window_idxs = window_idxs.reshape(-1, n_events.item(), window_idxs.shape[-1])[:, 0, :]
        onset = scene.sources[tab_idx].sequence.events[0].onset.timepoint
        offset = scene.sources[tab_idx].sequence.events[0].offset.timepoint
        # Get amplitude/spectrum information
        tf_grid = scene.sources[tab_idx].renderer.AM_and_filter.tf_grid.permute(0, 2, 1)
        mean_spectrum_over_time = torch.zeros(tf_grid.shape[0], tf_grid.shape[1])
        # Get the active timepoints that occur within the event boundaries
        for batch_idx in range(onset.shape[0]):
            this_window_ts = window_ts[batch_idx, :]
            this_window_idxs = window_idxs[batch_idx, :]
            if len(this_window_idxs) < len(this_window_ts):
                this_window_idxs[this_window_idxs >= len(this_window_ts)] = len(this_window_ts) - 1
                padded_ts = this_window_ts[this_window_idxs]
            elif len(this_window_idxs) == len(this_window_ts):
                padded_ts = this_window_ts
            active_timepoints = (onset[batch_idx] <= padded_ts)*(padded_ts <= offset[batch_idx])
            active_spectrum = tf_grid[batch_idx, :, active_timepoints]
            mean_spectrum_over_time[batch_idx, :] = active_spectrum.mean(1)
        spectrum = {
            "x": scene.sources[tab_idx].gps.spectrum.feature.gp_x.squeeze().detach().cpu().numpy()[0, :],
            "y": mean_spectrum_over_time.detach().cpu().numpy()
            }
        return spectrum
    return latents_logger


def get_explanation_from_inits(inference_dir, sound_name, sound_group, expt_name):
    """ Get the initialization for each stimulus in the experiment """

    expt_folders = glob(os.path.join(
        inference_dir, expt_name, sound_group, sound_name + "_seed*", "*", ""
        ))
    initializations = []
    for expt_folder in expt_folders:
        initializations.append(expt_folder.split(os.sep)[-2].split("-seed")[0])
    initializations = np.unique(initializations)
    print(f"Initializations for {sound_name}:", initializations)
    results_dict = {}

    for init_set in initializations:
        seed_initializations = glob(os.path.join(
            inference_dir, expt_name, sound_group, 
            sound_name + "_seed*", f"*{init_set}*/"
            ))
        print("n_seeds for {}: ".format(init_set), len(seed_initializations))
        for initialization in seed_initializations:
            seed_idx = int(initialization[-2])
            _, ckpt = inference.io.restore_hypothesis(initialization)
            if ckpt is None:
                continue
            # The zeroth source is always the tab source in the initializations
            latent_logger = define_latents_logger(0)
            scene = ckpt["metrics"].best_scene
            spectrum = latent_logger(scene)
            spectrum["y"] = spectrum["y"].mean(0)
            results_dict[(init_set, seed_idx)] = {
                "elbo": ckpt["metrics"].elbo(),
                "spectrum": spectrum
                }

    return results_dict


def get_sound_from_iterative(inference_dir, sound_name, sound_group, expt_name):

    expt_dir = os.path.join(inference_dir, expt_name, sound_group, sound_name, "")
    with open(expt_dir + "results.json", "r") as f:
        results = json.load(f)
    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons",
        "results", "gen-model", sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)

    round_idx = len(results["keep"]) - 1
    expl = int(results["keep"][-1][0])
    best_initialization = os.path.join(
        inference_dir, expt_name, sound_group, sound_name,
        f"round{round_idx:03d}-{expl:03d}", ""
        )
    _, ckpt = inference.io.restore_hypothesis(best_initialization)
    scene = ckpt["metrics"].best_scene

    batch_idx = ckpt["metrics"].best_scene_scores.argmax()
    for source_idx, source in enumerate(scene.sources):
        sf.write(
            os.path.join(
                sequential_folder,
                f"{sound_name}_round{round_idx:03d}-{expl:03d}_source{source_idx:02d}_estimate.wav"
                ),
            source.source_wave.detach().numpy()[batch_idx, :],
            scene.audio_sr
            )

    return


def compare_tabs(inferred_dict, comparison_dict):
    """ Compare inferred tabs and comparison stimuli """
    if not np.all(inferred_dict["x"] == inferred_dict["x"]):
        raise ValueError("spectra need to be sampled at the same frequencies.")
    return np.sqrt(np.mean(np.square(
        inferred_dict["y"] - comparison_dict["spectrum"]["y"]
        )))


def compare_tabs_from_multiple(results_dict, comparison_dicts, savepath, sound_name):
    """ Find the best spectrum match for a single sound """

    # Get keys for input dicts
    levels = sorted([k for k in comparison_dicts.keys()], key=lambda k: float(k))
    seeds = [seed_idx for (init, seed_idx) in results_dict.keys()]
    seeds = np.unique(seeds)

    # Loop over seeds, modal inits, and comparison stimuli
    distance_matrix = np.full((len(seeds), len(comparison_dicts)), np.nan)

    for seed in seeds:
        inits = [init for (init, seed_idx) in results_dict.keys() if seed_idx == seed]
        for comparison_idx, level in enumerate(levels):
            comparison_dict = comparison_dicts[level]
            cd_to_use = comparison_dict[(f'mid{int(level):03d}', seed)]

            elbos = []
            distances = []
            for init_idx, init in enumerate(inits):
                spectrum = results_dict[(init, seed)]["spectrum"]
                distances.append(compare_tabs(spectrum, cd_to_use))
                elbos.append(results_dict[(init, seed)]["elbo"])

            # Weight the distances by the elbos
            weights = np.exp(np.array(elbos) - scipy.special.logsumexp(elbos))
            weighted_distance = np.sum(weights * np.array(distances))
            distance_matrix[seed, comparison_idx] = weighted_distance

    level_floats = [float(level) for level in levels]
    levelarr = np.array(level_floats)
    level_idxs = np.argmin(distance_matrix, axis=1)
    best_match = np.nanmean(levelarr[level_idxs])
    stderr = np.nanstd(levelarr[level_idxs]) / np.sqrt(np.sum(~np.isnan(level_idxs)))

    return best_match, stderr, distance_matrix


def compute_comparisons(sound_names, inferred_dicts, comparison_dicts, shared_comparisons=True):
    """
    sounds: list[str] giving the name of the wave file
    inferred_dicts: dict[sound_name: inference_results]
    """
    results = np.full(len(sound_names), np.nan)
    stderr = np.full(len(sound_names), np.nan)
    for s_idx, sound_name in enumerate(sound_names):
        use_comparison_dicts = comparison_dicts if shared_comparisons else comparison_dicts[sound_name[1:]]
        results[s_idx], stderr[s_idx], distance_matrix, spectrum = compare_tabs_from_multiple(inferred_dicts[sound_name], use_comparison_dicts, savepath, sound_name)
    print([pair for pair in zip(sound_names, results)])
    return results, stderr


def figure1(sound_group, expt_name, inference_dir, sound_dir, savepath, make_new=False):

    # Get explanations from inference
    results_fn = savepath + "sc_expt1_results.npy"
    if not os.path.isfile(results_fn) or make_new:
        results_dict = {}
    else:
        results_dict = np.load(results_fn, allow_pickle=True).item()

    sound_names = ["1i", "1ii", "1iii", "1iv", "1v"]
    if "infer" not in results_dict.keys():
        results_dict["infer"] = {}
    for sound_name in sound_names:
        if sound_name not in results_dict["infer"].keys():
            print("Getting best scene for ", sound_name)
            results_dict["infer"][sound_name] = get_explanation_from_inits(
                inference_dir, "sc_" + sound_name, sound_group, expt_name
                )
            np.save(results_fn, results_dict)

    # Get explanations of comparison stimuli
    match_stimuli = glob(os.path.join(sound_dir, sound_group + "_match", "sc_1_*.wav"))
    # match_stimuli = match_stimuli+ glob("/om2/user/lbh/basatorch/sounds/" + sound_group + "_match/sc_1_*.wav")
    if "compare" not in results_dict.keys():
        results_dict["compare"] = {}
    for match_stimulus in match_stimuli:
        sound_name = match_stimulus.split(os.sep)[-1][:-10]
        mid_level = sound_name.split("_")[-1]
        if float(mid_level) not in results_dict["compare"].keys():
            results_dict["compare"][float(mid_level)] = get_explanation_from_inits(inference_dir, sound_name, sound_group + "_match", expt_name)
            np.save(results_fn, results_dict)

    # Get distance matrix between each sound and the cache
    inferred_dicts = results_dict["infer"]
    comparison_dict = results_dict["compare"]
    results, stderr = compute_comparisons(sound_names, inferred_dicts, comparison_dict)
    savepath = os.path.join(savepath, "sc_fig1_bar.png")
    fig1_plotter(sound_names, results, stderr, savepath, sound_group, expt_name)

    return results, stderr


def fig1_plotter(sound_names, results, stderr, savepath, sound_group, expt_name):

    fig1_human_means = [-12, 29, 7.5, -7.5, -11]
    fig1_human_stds = [0.71, 0.5, 1.4, 0.8, 0.8]

    x = list(range(len(sound_names)))
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5.5, 2))
    for ax_idx in [0, 1]:
        if ax_idx == 1:
            Yb = [r + 15 for r in results]
            Y = results
            S = stderr
        elif ax_idx == 0:
            Yb = [r+15 for r in fig1_human_means]
            Y = fig1_human_means
            S = fig1_human_stds
        axs[ax_idx].bar(x, Yb, bottom=-15.0, color="grey", tick_label=sound_names)
        axs[ax_idx].errorbar(x, Y, S, fmt=" ", ecolor="black", capsize=5, alpha=0.8)
        axs[ax_idx].set_ylim([-15, 35])
        axs[ax_idx].axhline(20.0, ls='--', color="black")
        axs[ax_idx].scatter([1], [30.0], color="black")

    axs[0].set_title("McDermott & Oxenham (2008): Expt. 1")
    axs[0].set_ylabel("Spectrum level of Comparison\n Middle Band (dB)")
    title = "/".join([sound_group, expt_name])
    axs[1].set_title(f"Genmodel:{title}", fontsize="xx-small")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath[:-3]+"svg")
    plt.close()


def figure2(sound_group, expt_name, inference_dir, sound_dir, savepath, make_new=False):

    results_fn = os.path.join(savepath, "sc_expt2_results.npy")
    if not os.path.isfile(results_fn) or make_new:
        results_dict = {}
    else:
        results_dict = np.load(results_fn, allow_pickle=True).item()

    sound_names = ["2i", "2ii", "2iii", "2iv", "2v", "2vi"]
    if "infer" not in results_dict.keys():
        results_dict["infer"] = {}
    for sound_name in sound_names:
        if sound_name not in results_dict["infer"].keys():
            results_dict["infer"][sound_name] = get_explanation_from_inits(inference_dir, "sc_" + sound_name, sound_group, expt_name)
            np.save(results_fn, results_dict)

    # Get explanations of comparison stimuli
    if "compare" not in results_dict.keys():
        results_dict["compare"] = {}
    for sound_name in sound_names:
        sn_key = sound_name[1:]
        if sn_key not in results_dict["compare"].keys():
            results_dict["compare"][sn_key] = {}
        match_stimuli = glob(os.path.join(
            sound_dir, sound_group + "_match", f"sc_{sound_name}_*.wav"
            ))
        for match_stimulus in match_stimuli:
            match_sound_name = match_stimulus.split(os.sep)[-1][:-10]
            mid_level = match_sound_name.split("_")[-1]
            # There is only one seed/init for the comparison stimuli
            if float(mid_level) not in results_dict["compare"][sn_key].keys():
                results_dict["compare"][sn_key][float(mid_level)] = get_explanation_from_inits(inference_dir, match_sound_name, sound_group + "_match", expt_name)
                np.save(results_fn, results_dict)

    # In experiment 2, compare each sound to only the comparison sounds with the same tabs.
    # Get distance matrix between each sound and the cache
    inferred_dicts = results_dict["infer"]
    comparison_dict = results_dict["compare"]
    results, stderr = compute_comparisons(sound_names, inferred_dicts, comparison_dict, shared_comparisons=False)

    savepath = os.path.join(savepath, "sc_fig2_bar.png")
    fig2_plotter(sound_names, results, stderr, savepath, sound_group, expt_name)

    return results, stderr


def fig2_plotter(sound_names, results, stderr, savepath, sound_group, expt_name):

    fig2_human_means = [2, 7, 10, 8, 4.5, 0]
    fig2_human_stds = [1.7, 1.0, 1.0, 1.32, 1.1, 1.1]

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5.5, 2))
    bottom = -6
    xs = [35.0, 30.0, 25.0, 20.0, 15.0, 10.0]
    for ax_idx in [0, 1]:
        axs[ax_idx].scatter(
            list(range(len(sound_names))), xs,
            color="black", marker="x", zorder=4, label="Masker level"
            )
        axs[ax_idx].scatter(
            list(range(len(sound_names))), [x - 10 for x in xs], 
            color="black", marker="o", zorder=3, label="Masker level - 10 dB"
            )
        axs[ax_idx].scatter(
            list(range(len(sound_names))), [x - 5 for x in xs[::-1]],
            facecolors='white', edgecolors='black', marker="o", zorder=2, label="Tab level"
            )
        if ax_idx == 0:
            Yb = [r - bottom for r in fig2_human_means]
            S = fig2_human_stds
            Y = fig2_human_means
        else:
            Yb = [r - bottom for r in results]
            S = stderr
            Y = results
        x = list(range(len(sound_names)))
        axs[ax_idx].bar(x, Yb, bottom=bottom, color="grey", tick_label=sound_names, zorder=1)
        axs[ax_idx].errorbar(x, Y, S, fmt=" ", ecolor="black", capsize=5, alpha=0.5)
    
    axs[0].legend(fontsize="small")
    axs[0].set_title("McDermott & Oxenham (2008): Expt. 2")
    title = "/".join([sound_group, expt_name])
    axs[1].set_title(f"Gen model:{title}", fontsize="xx-small")
    axs[0].set_ylim([bottom, 40])
    axs[0].set_ylabel("Spectrum level of Comparison\n Middle Band (dB)")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath[:-3]+"svg")
    plt.close()


if __name__ == '__main__':
    print("Analyzing spectral completion!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str, help="folder where wavs are saved")
    parser.add_argument("--expt-name", type=str, help="folder to group inference results together")
    parser.add_argument("--inference-dir", type=str, default=os.environ["inference_dir"], help="top-level folder for inference")
    parser.add_argument("--sound-dir", type=str, default=os.environ["sound_dir"], help="top-level folder for inference")
    parser.add_argument("--results-dir", type=str, default="home")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--model-comparison-dir", type=str, required=False)
    args = parser.parse_args()
    if args.results_dir == "home":
        results_dir = os.path.join(os.environ["inference_dir"], args.expt_name, args.sound_group, "")
    else:
        results_dir = args.results_dir

    fig1_results, fig1_stderr = figure1(args.sound_group, args.expt_name, args.inference_dir, args.sound_dir, results_dir, make_new=args.new)    
    fig2_results, fig2_stderr = figure2(args.sound_group, args.expt_name, args.inference_dir, args.sound_dir, results_dir, make_new=args.new)

    if args.model_comparison_dir is not None:
        result = {"fig1": fig1_results.tolist(), "fig2": fig2_results.tolist()}
        np.save(
            os.path.join(args.model_comparison_dir, args.sound_group, args.expt_name, "result.npy"),
            result,
            allow_pickle=True
            )
        for_plots = {
            "fig1": {"means": fig1_results.tolist(), "stderrs": fig1_stderr.tolist()},
            "fig2": {"means": fig2_results.tolist(), "stderrs": fig2_stderr.tolist()}
            }
        np.save(
            os.path.join(args.model_comparison_dir, args.sound_group, args.expt_name, "for_plots.npy"),
            for_plots,
            allow_pickle=True
            )
