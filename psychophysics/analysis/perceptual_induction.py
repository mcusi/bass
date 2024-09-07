import os
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf

import inference.io
import psychophysics.hypotheses.hutil as hutil
import psychophysics.analysis.thresholds as pat


def get_hypothesis_elbo(inference_dir, sound_name, sound_group, expt_name, expt_type, seed, total_iters_should_be=8000):
    """ Find the elbo for each hypothesis """

    hypotheses = ["present", "absent"] if expt_type == "masking" else ["discontinue", "continue"]

    hypothesis_path = os.path.join(inference_dir, expt_name, sound_group, sound_name, "")
    results = {}
    for hypothesis in hypotheses:
        hpath = hypothesis + f"-seed{seed}"
        try:
            with open(hypothesis_path + hpath + "/metrics.json", "r") as f:
                metrics = json.load(f)
                if metrics["total_iters"] < total_iters_should_be:
                    print(f"{hypothesis_path+hpath} crashed")
                    results[hypothesis] = np.nan
                    continue
                results[hypothesis] = metrics["elbo"]
        except:
            results[hypothesis] = np.nan

    # Return results, what you expect at higher levels (present/discontinue)
    # before lower levels (absent, continue)
    if expt_type == "masking":
        return [results["present"], results["absent"]] 
    elif expt_type == "continuity":
        return [results["discontinue"], results["continue"]]


def get_sound_from_iterative(inference_dir, sound_name, sound_group, expt_name):

    expt_dir = os.path.join(inference_dir, expt_name, sound_group, sound_name, "")
    with open(expt_dir + "results.json", "r") as f:
        results = json.load(f)
    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons", "results", "gen-model", sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)

    round_idx = len(results["keep"])-1
    hypothesis_idx = 0
    expl = int(results["keep"][-1][hypothesis_idx])

    best_initialization = os.path.join(
        inference_dir, expt_name, sound_group, sound_name, f"round{round_idx:03d}-{expl:03d}", ""
        )
    _, ckpt = inference.io.restore_hypothesis(best_initialization)
    scene = ckpt["metrics"].best_scene

    print(f"~~ Using hypothesis {expl} ~~")
    batch_idx = (scene.ll + scene.lp - scene.lq).argmax().item()
    for source_idx, source in enumerate(scene.sources):
        sf.write(
            os.path.join(sequential_folder, f"{sound_name}_round{round_idx:03d}-{expl:03d}_source{source_idx:02d}_estimate.wav"),
            source.source_wave.detach().numpy()[batch_idx, :],
            scene.audio_sr
            )

    return


def compute_threshold(odds, tone_levels):
    try:
        only_keep = ~np.isnan(odds)
        odds = np.array(odds)[only_keep]
        tone_levels = np.array(tone_levels)[only_keep]
        boundary = pat.average_crossing_points_log(tone_levels, odds)
        return boundary
    except:
        return np.nan


def get_results(inference_dir, sound_group, expt_name, settings, n_seeds):

    n_freqs = len(settings["tone_freqs"])
    n_levels = len(settings["tone_levels"])
    results = {
        "masking": np.full((n_freqs, n_levels, n_seeds, 2), np.nan),
        "continuity": np.full((n_freqs, n_levels, n_seeds, 2), np.nan)
        }
    for expt_type in ["masking", "continuity"]:
        for f0_idx, f0 in enumerate(settings["tone_freqs"]):
            for dB_idx, dB in enumerate(settings["tone_levels"]):
                for seed_idx in range(n_seeds):
                    sound_name = f"{expt_type}_f{f0:04d}_l{dB:03d}_seed{seed_idx}"
                    hypothesis_elbos = get_hypothesis_elbo(
                        inference_dir, sound_name, sound_group,
                        expt_name, expt_type, seed=seed_idx
                        )
                    results[expt_type][f0_idx, dB_idx, seed_idx, :] = hypothesis_elbos

    return results


def multiple(sound_group, expt_name, inference_dir, sound_dir, results_path, n_seeds, model_comparison_dir=None):

    try:
        settings = np.load(os.path.join(
                sound_dir, sound_group, "pi_expt3_settings_seed0.npy"
            ), allow_pickle=True).item()
    except:
        settings = np.load(os.path.join(
                sound_dir, sound_group, "pi_expt3_settings.npy"
            ), allow_pickle=True).item()

    results = get_results(inference_dir, sound_group, expt_name, settings, n_seeds)

    cmap = hutil.categorical_cmap(len(settings["tone_freqs"]), n_seeds)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)

    masking_thresholds = np.full((len(settings["tone_freqs"]), n_seeds), np.nan)
    continuity_thresholds = np.full((len(settings["tone_freqs"]), n_seeds), np.nan)
    for seed_idx in range(n_seeds):
        for f0_idx, f0 in enumerate(settings["tone_freqs"]):
            masking_odds = results["masking"][f0_idx, :, seed_idx, 0] - results["masking"][f0_idx, :, seed_idx, 1]
            continuity_odds = results["continuity"][f0_idx, :, seed_idx, 0] - results["continuity"][f0_idx, :, seed_idx, 1]
            masking_thresholds[f0_idx, seed_idx] = compute_threshold(masking_odds, settings["tone_levels"])
            continuity_thresholds[f0_idx, seed_idx] = compute_threshold(continuity_odds, settings["tone_levels"])

    mean_masking_thresholds = np.nanmean(masking_thresholds, axis=1)
    mean_continuity_thresholds = np.nanmean(continuity_thresholds, axis=1)

    masking_non_nan = np.sum(~np.isnan(masking_thresholds), axis=1)
    stderr_m = np.where(
        masking_non_nan > 0,
        np.nanstd(masking_thresholds, axis=1)/np.sqrt(masking_non_nan),
        np.zeros(len(masking_non_nan))
        )

    continuity_non_nan = np.sum(~np.isnan(continuity_thresholds), axis=1)
    stderr_c = np.where(
        continuity_non_nan > 0,
        np.nanstd(continuity_thresholds, axis=1)/np.sqrt(continuity_non_nan),
        np.zeros(len(continuity_non_nan))
        )

    M = [27.5, 33, 21, 40, 42.5]
    C = [15, 19, 4, 19, 21]
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(5.8, 2.2))
    axs[0].plot(
        np.asarray(settings["tone_freqs"]), M,
        "ko-", markerfacecolor="white", label="Masking"
        )
    axs[0].plot(
        np.asarray(settings["tone_freqs"]), C,
        "ko-", markeredgecolor=(0, 0, 0, 1.0),
        markerfacecolor=(0, 0, 0, 0.5), alpha=0.7,
        label="Continuity"
        )
    axs[1].errorbar(
        np.asarray(settings["tone_freqs"]),
        mean_masking_thresholds, stderr_m,
        fmt="ko-", markerfacecolor="white", label="Masking"
        )
    axs[1].errorbar(
        np.asarray(settings["tone_freqs"]),
        mean_continuity_thresholds, stderr_c,
        fmt="ko-", markeredgecolor=(0, 0, 0, 1.0),
        markerfacecolor=(0, 0, 0, 0.5),
        alpha=0.7, label="Continuity"
        )
    axs[0].legend(fontsize="small")
    axs[0].set_ylabel("Threshold\nabove audibility (dB)")
    axs[0].set_ylim([0, 50])
    axs[1].set_ylim([40, 90])
    axs[1].set_ylabel("Threshold (dB)")
    axs[1].set_xlabel("Tone frequency (Hz)")
    axs[0].set_xlabel("Tone frequency (Hz)")
    axs[0].set_title("Warren et al., 1972")
    axs[1].set_title(f"Gen model:{sound_group} {expt_name}".format(sound_group, expt_name), fontsize="x-small")
    axs[1].semilogx()
    axs[1].set_xscale('log')
    axs[1].set_xticks([250, 500, 1000, 2000, 4000])
    axs[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    print(os.path.join(results_path, f"induction_thresholds_{expt_name}_{sound_group}.png"))
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"induction_thresholds_{expt_name}_{sound_group}.png"))
    plt.savefig(os.path.join(results_path, f"induction_thresholds_{expt_name}_{sound_group}.svg"))
    plt.close()

    if model_comparison_dir is not None:
        model_results = {
            "masking": mean_masking_thresholds.tolist(),
            "continuity":mean_continuity_thresholds.tolist()
            }
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result.npy"),
            model_results, allow_pickle=True
            )
        for_plots = {
            "freqs": settings["tone_freqs"],
            "mean_m": mean_masking_thresholds,
            "stderr_m": stderr_m,
            "mean_c":mean_continuity_thresholds,
            "stderr_c":stderr_c
            }
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots.npy"),
            for_plots, allow_pickle=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str,
                        help="folder where wavs saved")
    parser.add_argument("--expt-name", type=str,
                        help="folder that groups inference results together")
    parser.add_argument("--inference-dir", type=str,
                        default=os.environ["inference_dir"],
                        help="top-level folder for inference")
    parser.add_argument("--sound-dir", type=str,
                        default=os.environ["sound_dir"],
                        help="top-level folder for sounds")
    parser.add_argument("--results-dir", type=str, default="home")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--model-comparison-dir", type=str, required=False)
    args = parser.parse_args()
    if args.results_dir == "home":
        results_dir = os.path.join(
            args.inference_dir,
            args.expt_name,
            args.sound_group,
            ""
            )
    else:
        results_dir = args.results_dir
    multiple(
        args.sound_group,
        args.expt_name,
        args.inference_dir,
        args.sound_dir,
        results_dir,
        args.n_seeds,
        model_comparison_dir=args.model_comparison_dir
        )
