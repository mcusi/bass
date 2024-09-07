import os
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import inference.io
import psychophysics.analysis.thresholds as pat


def get_sound_from_iterative(inference_dir, sound_name, sound_group, expt_name):
    """ Get sound from sequential inference """

    expt_dir = os.path.join(inference_dir, expt_name, sound_group, sound_name, "")
    with open(expt_dir + "results.json", "r") as f:
        results = json.load(f)
    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons", "results", "gen-model", sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)

    round_idx = len(results["keep"]) - 1
    expl = int(results["keep"][-1][0])

    best_initialization = os.path.join(
        inference_dir, expt_name, sound_group, sound_name, f"round{round_idx:03d}-{expl:03d}", ""
        )
    _, ckpt = inference.io.restore_hypothesis(best_initialization)
    scene = ckpt["metrics"].best_scene

    batch_idx = ckpt["metrics"].best_scene_scores.argmax()
    for source_idx, source in enumerate(scene.sources):
        sf.write(
            sequential_folder + f"{sound_name}_round{round_idx:03d}-{expl:03d}_source{source_idx:02d}_estimate.wav",
            source.source_wave.detach().numpy()[batch_idx, :],
            scene.audio_sr
            )

    return


def get_hypothesis_elbo(inference_dir, sound_name, sound_group, expt_name, seed):
    """ Get marginal likelihood for each hypothesis """
    hypotheses = ["h", "hw", "hwq"]
    hypothesis_path = os.path.join(inference_dir, expt_name, sound_group, sound_name, "")
    results = {}
    for hypothesis in hypotheses:
        path = hypothesis_path + hypothesis + f"-seed{seed}/"
        try:
            with open(os.path.join(path, "metrics.json"), "r") as f:
                metrics = json.load(f)
                results[hypothesis] = metrics["elbo"]
        except:
            results[hypothesis] = np.nan
    return [max(results["hwq"], results["hw"]), results["h"]]


def compute_threshold(odds, mistuned_percents):
    only_keep = ~np.isnan(odds)
    odds = np.array(odds)[only_keep]
    if np.all(odds < 0):
        return max(mistuned_percents)+10
    mistuned_percents = np.array(mistuned_percents)[only_keep]
    boundary = pat.average_crossing_points_log(mistuned_percents, odds)
    return boundary


def psychophysics(sound_group, expt_name, inference_dir, sound_dir, results_path, n_seeds, model_comparison_dir=None):

    settings = np.load(os.path.join(
        sound_dir, sound_group, "mh_expt_settings.npy"
        ), allow_pickle=True).item()
    n_f0s = len(settings["fundamentals"])
    n_durs = len(settings["durations"])
    n_midxs = len(settings["mistuned_idxs"])
    n_pvs = len(settings["mistuned_percents"])
    results = np.full((n_durs, n_f0s, n_midxs, n_pvs, n_seeds, 2), np.nan)
    lev = settings["mistuned_level"]
    for midx, mistuned_idx in enumerate(settings["mistuned_idxs"]):
        for fidx, harmonic_f0 in enumerate(settings["fundamentals"]):
            for didx, duration_ms in enumerate(settings["durations"]):
                for pidx, p_val in enumerate(settings["mistuned_percents"]):
                    for sidx in range(n_seeds):
                        sound_name = f"f{harmonic_f0:04d}_dur{duration_ms}_harm{mistuned_idx}_p{p_val}_dB{lev}"
                        results[didx, fidx, midx, pidx, sidx, :] = get_hypothesis_elbo(
                            inference_dir, sound_name, sound_group, expt_name, sidx
                            )
                        if np.isnan(results[didx, fidx, midx, pidx, sidx, 0]):
                            print(f"Is nan: dur={settings['durations'][didx]}, f0={settings['fundamentals'][fidx]} misIdx={settings['mistuned_idxs'][midx]} seed={sidx} percent={p_val} hyp=0")
                        if np.isnan(results[didx, fidx, midx, pidx, sidx, 1]):
                            print(f"Is nan: dur={settings['durations'][didx]}, f0={settings['fundamentals'][fidx]} misIdx={settings['mistuned_idxs'][midx]} seed={sidx} percent={p_val} hyp=1")


    for duridx, duration_ms in enumerate(settings["durations"]):
        fig, axs = plt.subplots(len(settings["fundamentals"]),1, sharex=True)
        for idx, harmonic_f0 in enumerate(settings["fundamentals"]):
            for menumidx, mistuned_idx in enumerate(settings["mistuned_idxs"]):
                for sidx in range(n_seeds):
                    x = settings["mistuned_percents"]
                    log_odds = results[duridx, idx, menumidx, :, sidx, 0] - results[duridx, idx, menumidx, :, sidx, 1]
                    axs[idx].plot(x, log_odds, alpha=0.8)
                    plt.savefig(results_path + "mistuned_dur{}_logodds.png".format(duration_ms))
            axs[idx].set_title("F0={}Hz".format(harmonic_f0))
            if idx == 0:
                axs[idx].legend(settings["mistuned_idxs"], fontsize="x-small")
            if idx == len(settings["fundamentals"]) - 1:
                axs[idx].set_xlabel("Mistuned percent")
                axs[idx].set_ylabel("h <- Log odds -> w+h")
        plt.tight_layout()
        print(results_path + "mistuned_dur{}_logodds.png".format(duration_ms))
        plt.savefig(results_path + "mistuned_dur{}_logodds.png".format(duration_ms))
        plt.close()

    thresholds = np.full((n_durs, n_f0s, n_midxs, n_seeds), np.nan)
    for dur_idx in range(len(settings["durations"])):
        for f0_idx in range(len(settings["fundamentals"])):
            for mis_idx in range(len(settings["mistuned_idxs"])):
                for seed_idx in range(n_seeds):
                    log_odds = results[dur_idx, f0_idx, mis_idx, :, seed_idx, 0] - results[dur_idx, f0_idx, mis_idx, :, seed_idx, 1]
                    if np.any(np.isnan(log_odds)):
                        continue
                    thresholds[dur_idx, f0_idx, mis_idx, seed_idx] = compute_threshold(log_odds, settings["mistuned_percents"])

    # All at 400 ms
    human_results = {
        (100, 1): [5, 4, 20, 5],
        (100, 2): [2.5, 2.0, 5.0, 2.1],
        (100, 3): [1.8, 4, 4, 1.0],
        (200, 1): [2.0, 1.8, 4.6, 1.4],
        (200, 2): [2.0, 1.9, 2.4, 1.3],
        (200, 3): [1.4, 1.8, 2.0, 1.3],
        (400, 1): [2.0, 2.1, 2.5, 0.8],
        (400, 2): [1.4, 2.0, 1.5, 0.8],
        (400, 3): [0.8, 1.9, 2.5, 1.0]
        }

    duration_idx = [i for i in range(len(settings["durations"])) if settings["durations"][i] == 400][0]
    TT = thresholds[duration_idx, :, :, :]
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5.2, 2.2))
    ax = axs[0]
    x = 0.4*np.arange(len(settings["fundamentals"]))
    width = 0.1

    # First plot human results
    ax.bar(
        x - width, [np.mean(human_results[(_f, 1)]) for _f in [100, 200, 400]],
        width, color="whitesmoke", edgecolor="black",
        label='Harmonic=1'
        )
    ax.errorbar(
        x - width, [np.mean(human_results[(_f, 1)]) for _f in [100, 200, 400]],
        [np.std(human_results[(_f, 1)])/2 for _f in [100, 200, 400]],
        fmt=" ", ecolor="black", capsize=5
        )
    ax.bar(
        x,  [np.mean(human_results[(_f, 2)]) for _f in [100, 200, 400]],
        width, color="silver", edgecolor="black",
        label='Harmonic=2'
        )
    ax.errorbar(
        x, [np.mean(human_results[(_f, 2)]) for _f in [100, 200, 400]],
        [np.std(human_results[(_f, 2)])/2 for _f in [100, 200, 400]],
        fmt=" ", ecolor="black", capsize=5
        )
    ax.bar(
        x + width, [np.mean(human_results[(_f, 3)]) for _f in [100, 200, 400]],
        width, color="dimgrey", edgecolor="black", label='Harmonic=3'
        )
    ax.errorbar(
        x + width, [np.mean(human_results[(_f, 3)]) for _f in [100, 200, 400]],
        [np.std(human_results[(_f,3)])/2 for _f in [100, 200, 400]],
        fmt=" ", ecolor="black", capsize=5
        )
    ax.set_title("Moore et al. (1986)")
    # Plot model results
    ax = axs[-1]
    ax.bar(
        x - width,  np.nanmean(TT[:, 0, :], axis=-1),
        width,  color="whitesmoke", edgecolor="black",
        label='Harmonic=1'
        )
    ax.bar(
        x, np.nanmean(TT[:, 1, :],axis=-1),
        width, color="silver", edgecolor="black",
        label='Harmonic=2'
        )
    ax.bar(
        x + width, np.nanmean(TT[:, 2, :], axis=-1),
        width, color="dimgrey", edgecolor="black",
        label='Harmonic=3'
        )
    ax.errorbar(
        x - width, np.nanmean(TT[:, 0, :], axis=-1),
        np.nanstd(TT[:, 0, :],axis=-1)/np.sqrt(np.count_nonzero(~np.isnan(TT[:, 0, :]), axis=-1)),
        fmt=" ", ecolor="black", capsize=5
        )
    ax.errorbar(
        x, np.nanmean(TT[:, 1, :], axis=-1),
        np.nanstd(TT[:, 1, :], axis=-1)/np.sqrt(np.count_nonzero(~np.isnan(TT[:, 1, :]), axis=-1)),
        fmt=" ", ecolor="black", capsize=5
        )
    ax.errorbar(
        x + width, np.nanmean(TT[:, 2, :], axis=-1),
        np.nanstd(TT[:, 2, :], axis=-1)/np.sqrt(np.count_nonzero(~np.isnan(TT[:, 2, :]), axis=-1)),
        fmt=" ", ecolor="black", capsize=5
        )
    ax.set_xticks(x, labels=[str(f) for f in settings["fundamentals"]])
    ax.set_title(f"Genmodel:{sound_group} {expt_name}", fontsize="small")

    axs[0].legend(fontsize="small")
    plt.semilogy()
    axs[0].set_yticks(
        [0.2, 0.5, 1.0, 2.0, 5.0, 10., 20., 40.],
        labels=[0.2, 0.5, 1.0, 2.0, 5.0, 10., 20., 40.]
        )
    axs[0].set_xlabel("Fundamental frequency (Hz)")
    axs[0].set_ylabel("Threshold (% fundamental)")
    axs[0].set_ylim([0.2, max(settings["mistuned_percents"])])
    plt.tight_layout()
    print(os.path.join(results_path, "compare_to_human.png"))
    plt.savefig(os.path.join(results_path, "compare_to_humans.png"))
    plt.savefig(os.path.join(results_path, "compare_to_humans.pdf"))
    plt.close()

    if model_comparison_dir is not None:
        model_results = {}
        for f0_idx, f0 in enumerate(settings["fundamentals"]):
            for m_arr_idx, m_idx in enumerate(settings["mistuned_idxs"]):
                model_results[(f0, m_idx)] = np.nanmean(TT[f0_idx, m_arr_idx])
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result.npy"),
            model_results, allow_pickle=True
            )
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots.npy"),
            TT, allow_pickle=True
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str,
                        help="folder where wavs are saved")
    parser.add_argument("--expt-name", type=str,
                        help="folder with grouped inference results")
    parser.add_argument("--inference-dir", type=str,
                        default=os.environ["inference_dir"],
                        help="top-level folder for inference")
    parser.add_argument("--sound-dir", type=str, default=os.environ["sound_dir"],
                        help="top-level folder for sounds")
    parser.add_argument("--results-dir", type=str, default="home")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--model-comparison-dir", type=str, required=False)
    args = parser.parse_args()
    if args.results_dir == "home":
        results_dir = os.path.join(
            args.inference_dir, args.expt_name, args.sound_group, ""
            )
    else:
        results_dir = args.results_dir
    psychophysics(
        args.sound_group,
        args.expt_name,
        args.inference_dir,
        args.sound_dir,
        results_dir,
        args.n_seeds,
        model_comparison_dir=args.model_comparison_dir
        )
