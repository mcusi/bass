import os
import json
import math
import numpy as np
from glob import glob
import soundfile as sf
import scipy.special
import matplotlib
import matplotlib.pyplot as plt

import inference.io

# Define reddish-grey colour
nan_color = [c/256. for c in [128, 128, 128]]


def get_sound_from_iterative(inference_dir, sound_name, sound_group, expt_name):

    # Get sequential inference results
    expt_dir = os.path.join(inference_dir, expt_name, sound_group, sound_name, "")
    with open(os.path.join(expt_dir, "results.json"), "r") as f:
        results = json.load(f)
    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons", "results", "gen-model",
        sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)

    # Retrieve results
    round_idx = len(results["keep"]) - 1
    expl = int(results["keep"][-1][0])
    best_initialization = os.path.join(
        inference_dir, expt_name,
        sound_group, sound_name, f"round{round_idx:03d}-{expl:03d}", ""
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


def read_json(hypothesis_path, seeds, filename="metrics"):
    results = np.full((len(seeds),), np.nan)
    paths = glob(hypothesis_path + "-seed*/")
    for p_idx, path in enumerate(paths):
        if p_idx >= len(seeds):
            continue
        print(path)
        if os.path.isfile(path + filename + ".json"):
            with open(path + filename + ".json", "r") as f:
                x = json.load(f)
            results[p_idx] = x["elbo"]
    print(results)
    return results


def plot_van_noorden(M, delta_frequency, delta_time, sound_group=None, expt_name=None, results_dir=None, model_comparison_dir=None):
    """ Make results plot for ABA bistability experiment"""

    # Get comparison
    # ABA advantage over AAA
    # 0=galloping-AAA-one-coherence, 1=isoch-ABA-2-fission
    log_odds = M[:, :, 1, :] - M[:, :, 0, :]
    log_odds = np.nanmean(log_odds, axis=-1)
    print("Log odds (bistability, ABA-AAA): ", log_odds)

    # Uses data from Van Noorden
    cmap = copy(matplotlib.cm.get_cmap("bwr"))
    mycolormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycolormap", [cmap(0.0), cmap(0.4), cmap(0.5), cmap(0.6), cmap(1.0)]
        )
    # Stimulus timing
    t_h = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # milliseconds
    fusion_boundary = [2.9, 3, 3.1, 3, 4, 3.5, 4, 3, 3, 4]  # semitones
    temporal_coherence_boundary = [4, 4, 4.5, 4.8, 6, 7, 8, 9.5, 10.5, 12]  # semitones
    plt.figure(figsize=(4, 4))
    plt.plot(
        t_h, fusion_boundary,
        "ko-", markerfacecolor="white", label="Two-sources boundary"
        )
    plt.plot(
        t_h, temporal_coherence_boundary,
        "ko-", label="One-source boundary"
        )
    DTs = []
    DFs = []
    CS = []
    for f_idx, df in enumerate(delta_frequency):
        for t_idx, dt in enumerate(delta_time):
            DTs.append(dt)
            DFs.append(df)
            CS.append(log_odds[f_idx, t_idx])
    plt.scatter(
        DTs, DFs, c=CS,
        edgecolors="black", alpha=0.5, s=150, cmap=mycolormap,
        vmin=-np.nanmax(np.abs(log_odds)), vmax=np.nanmax(np.abs(log_odds))
        )
    plt.ylim([0, 16])
    cbar = plt.colorbar()
    cbar.set_label("Log odds ('Two sources'-'One source')")
    plt.ylabel("Frequency difference (semitones, $\Delta f$)")
    plt.xlabel("Onset-to-onset time (milliseconds, $\Delta t$)")
    plt.title("Boundaries from van Noorden (1975)")
    plt.legend()
    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "aba.png"))
        plt.savefig(os.path.join(results_dir, "aba.svg"))
        np.save(os.path.join(results_dir, "aba.npy"), log_odds)
    plt.close()

    if model_comparison_dir is not None:
        model_results = {"tf": [], "d": []}
        for f_idx, df in enumerate(delta_frequency):
            for t_idx, dt in enumerate(delta_time):
                model_results["tf"].append((dt, df))
                model_results["d"].append(log_odds[f_idx, t_idx])
        for_plots = {"f": delta_frequency, "t": delta_time, "log_odds": log_odds}
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result_aba.npy"),
            model_results,
            allow_pickle=True
            )
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots_aba.npy"),
            for_plots
            )


def plot_cumulative(M, n_reps, sound_group, expt_name, cap=-2, savepath=None, results_dir=None, model_comparison_dir=None):
    """ Plot to report on cumulative repetition experiment """
    fig, axs = plt.subplots(1, 2, sharex=False, figsize=(6.0, 2.0))

    # Thompson results
    human_x = np.array([2, 3, 4, 5, 6]) - 1.5
    human_df8 = np.array([1.17, 1.5, 1.56, 1.67, 1.7])
    human_df8_stds = np.array([1.24, 1.56, 1.62, 1.72, 1.74]) - human_df8
    human_df4 = np.array([1.03, 1.06, 1.1, 1.16, 1.19])
    human_df4_stds = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
    # Plot human results
    axs[0].errorbar(
        human_x[:cap], human_df4[:cap], human_df4_stds[:cap],
        fmt="ko-", ms=5, markerfacecolor="white", capsize=5,
        label="$\Delta$f=4"
        )
    axs[0].errorbar(
        human_x[:cap], human_df8[:cap], human_df8_stds[:cap],
        fmt="ko-", ms=5,
        markeredgecolor=(0, 0, 0, 1.0), markerfacecolor=(0, 0, 0, 1.0), capsize=5,
        label="$\Delta$f=8"
        )
    axs[0].plot(
        [0, 3.25], [1.5, 1.5],
        "k--", alpha=0.5,
        label="Equal response rate (1 or 2)"
        )
    axs[0].set_ylim([1, 2.0])
    axs[0].set_ylabel("Average number of\n sources reported")
    axs[0].set_xlabel("Time from first participant response (s)")
    axs[0].set_title("Thompson et al., 2011")

    # Plot model results
    model_df4 = (M[:, 0, :, 1, :] - M[:, 0, :, 0, :]).mean(-1)[:, 0]
    model_df4_stds = (M[:, 0, :, 1, :] - M[:, 0, :, 0, :]).std(-1)[:, 0]
    model_df8 = (M[:, 0, :, 1, :] - M[:, 0, :, 0, :]).mean(-1)[:, 1]
    model_df8_stds = (M[:, 0, :, 1, :] - M[:, 0, :, 0, :]).std(-1)[:, 1]
    axs[1].errorbar(
        0.5*np.array(n_reps), model_df4, model_df4_stds,
        fmt="ko-",ms=5, markerfacecolor="white", capsize=5,
        label="$\Delta$f=4"
        )
    axs[1].errorbar(
        0.5*np.array(n_reps), model_df8, model_df8_stds,
        fmt="ko-", ms=5,
        markeredgecolor=(0, 0, 0, 1.0), markerfacecolor=(0, 0, 0, 1.0), capsize=5,
        label="$\Delta$f=8"
        )
    axs[1].plot(
        [0, 3.25], [0, 0],
        "k--", alpha=0.5,
        label="Hypotheses equally likely"
        )
    axs[1].set_ylabel("Log odds\n('Two-sources' - 'One-source')")
    axs[1].set_xlabel("Time from beginning of stimulus (s)")
    axs[0].legend(fontsize="small")
    axs[1].legend(fontsize="small")
    x0, x1 = axs[0].get_xlim()
    axs[0].set_xlim([0, 3.25])
    axs[1].set_xlim([0, 3.25])
    plt.tight_layout()
    if savepath is None:
        plt.savefig(
            os.path.join(results_dir, f"{expt_name}_{sound_group}_cumulativeh.png")
            )
        plt.savefig(
            os.path.join(results_dir, f"{expt_name}_{sound_group}_cumulativeh.svg")
            )
        plt.close()
    else:
        print(savepath)
        plt.savefig(savepath)
    plt.close()

    if model_comparison_dir is not None:
        model_results = {"tfr": [], "d": []}
        for rep_idx, rep in enumerate(n_reps):
            model_results["tfr"].append((125, 4, rep))
            model_results["d"].append(model_df4[rep_idx])
        for rep_idx, rep in enumerate(n_reps):
            model_results["tfr"].append((125, 8, rep))
            model_results["d"].append(model_df8[rep_idx])
        for_plots = {
            "df4":model_df4,
            "df4_stds": model_df4_stds,
            "df8":model_df8,
            "df8_stds": model_df8_stds,
        }
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result_cumul.npy"),
            model_results,
            allow_pickle=True
            )
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots_cumul.npy"),
            for_plots,
            allow_pickle=True
            )


def plot_captor(M, conditions, sound_group, expt_name, savepath=None, results_dir=None, model_comparison_dir=None):
    """ Plots for captor context experiment """

    # 3S or C+D advantage over D+T
    M[np.isnan(M)] = -np.inf
    log_odds = scipy.special.logsumexp(M[:, 1:, :], axis=1) - M[:, 0, :]
    lo_stderr = np.nanstd(log_odds, axis=-1)/np.sqrt(log_odds.shape[-1])
    log_odds = np.nanmean(log_odds, axis=-1)
    print(log_odds)

    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(6.0, 2.0))
    axs[0].scatter(range(4), [.65, .65, .76, .82], marker=".", color="black")
    axs[0].set_title("Bregman & Rudnicky (1975)")
    axs[0].set_ylabel("Proportion correct")
    axs[0].set_ylim([0, 1])
    x0, x1 = axs[0].get_xlim()
    axs[0].plot(
        [-1, 5], [0.5, 0.5],
        linestyle="--", color="black", alpha=0.5,
        label="Chance"
        )
    axs[0].set_xlim([x0, x1])
    axs[1].errorbar(
        range(4), log_odds, lo_stderr,
        fmt="k.", ecolor="black", capsize=5
        )
    axs[1].set_ylim([-np.max(log_odds)-3, np.max(log_odds)+3])
    axs[1].set_ylabel("Log odds\n(Target alone - \ntarget with distractors)")
    axs[1].set_xticks(range(4))
    axs[1].set_xticklabels(["None", "590 Hz", "1030 Hz", "1460 Hz"])
    x0, x1 = axs[0].get_xlim()
    axs[1].plot(
        [-1, 5], [0, 0],
        linestyle="--", color="black", alpha=0.5,
        label="Hypotheses equally probable"
        )
    axs[1].set_ylim([-10, 20])
    axs[0].set_ylim([0.2, 1.0])
    axs[1].set_xlabel("Captor condition")
    axs[0].legend(loc="lower left")
    axs[1].legend(loc="lower left")
    plt.tight_layout()
    if savepath is None:
        savepath = os.path.join(results_dir, "captor.png")
    plt.savefig(savepath)
    plt.savefig(savepath[:-4] + ".svg")
    plt.close()

    if model_comparison_dir is not None:
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result_captor.npy"),
            {"d": log_odds, "conds": ["None", "590 Hz", "1030 Hz", "1460 Hz"]},
            allow_pickle=True
            )
        for_plots = {"mean_log_odds": log_odds, "stderr": lo_stderr}
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots_captor.npy"),
            for_plots,
            allow_pickle=True
            )


def nanlogsumexp(x, axis):
    c = np.nanmax(x)
    return c + np.log(np.nansum(np.exp(x - c),axis=axis))


def plot_compete_average(M, hypotheses, sound_fns, sound_group, expt_name, savepath=None, results_dir=None, model_comparison_dir=None):
    """ Plot results figure for compete context experiemnt
        M: sound_file by hypothesis
            first half of sound files - absorb
            "AB-XY" - "AX-BY"
    """

    # Get model results
    together_hypotheses = np.array([
        (h[0] == h[1]) and (h[2] != h[0]) and (h[3] != h[0])
        for h in hypotheses
        ])
    isolate_sound_idxs = ["isolate" in sound_fn for sound_fn in sound_fns]
    absorb_sound_idxs = ["absorb" in sound_fn for sound_fn in sound_fns]
    # X-axis: absorb sounds vs. isolate sounds; y-axis difference in ELBO
    isolate_sound_elbos = M[isolate_sound_idxs, :, :]
    absorb_sound_elbos = M[absorb_sound_idxs, :, :]
    isolate_sound_together_elbo = nanlogsumexp(isolate_sound_elbos[:, together_hypotheses, :], axis=1)
    isolate_sound_separate_elbo = nanlogsumexp(isolate_sound_elbos[:, ~together_hypotheses, :], axis=1)
    isolate_sound_log_odds = isolate_sound_together_elbo - isolate_sound_separate_elbo
    absorb_sound_together_elbo = nanlogsumexp(absorb_sound_elbos[:, together_hypotheses, :], axis=1)
    absorb_sound_separate_elbo = nanlogsumexp(absorb_sound_elbos[:, ~together_hypotheses, :], axis=1)
    absorb_sound_log_odds = absorb_sound_together_elbo - absorb_sound_separate_elbo
    all_log_odds = np.concatenate((isolate_sound_log_odds, absorb_sound_log_odds), axis=0)
    stderr_log_odds = np.nanstd(all_log_odds, axis=-1)/np.sqrt(all_log_odds.shape[-1])
    mean_log_odds = np.nanmean(all_log_odds ,axis=-1)
    xlabels = ["Isolate"]*len(isolate_sound_log_odds)+["Absorb"]*len(absorb_sound_log_odds)

    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(6.0, 2.0))
    # Human results
    axs[0].scatter(
        range(7), [12.03, 12.69, 13.16, 13.76, 3.71, 3.84, 5.21],
        color="black", marker="."
        )
    axs[0].set_title("Bregman (1978)")
    axs[0].set_ylabel("Rating")
    axs[0].set_ylim([1, 14])
    axs[0].set_yticks([1, 4, 7, 11, 14]) 
    axs[0].set_yticklabels([
        "0:'Can't hear\nAB pair'",
        "",
        "7:Uncertain",
        "",
        "14:'Can hear\nAB pair'"
        ])
    x0, x1 = axs[0].get_xlim()
    axs[0].plot(
        [-1, 9], [7, 7],
        linestyle="--", color="black", alpha=0.5
        )
    axs[0].set_xlim([x0, x1])
    axs[1].scatter(
        range(7), mean_log_odds,
        color="black", marker="."
    )
    axs[1].errorbar(
        range(7), mean_log_odds, stderr_log_odds,
        fmt=" ", ecolor="black", capsize=5
        )
    axs[1].set_ylabel("Log odds\n(AB in own source - \n AB in separate sources)")
    axs[1].set_xticks(range(7))
    axs[1].set_xticklabels(
        [x+str(i) for i, x in enumerate(xlabels)],
        rotation=30, rotation_mode='anchor', ha="right"
        )
    axs[0].set_xticklabels(
        [x+str(i) for i, x in enumerate(xlabels)],
        rotation=30, rotation_mode='anchor', ha="right"
        )
    x0, x1 = axs[0].get_xlim()
    axs[1].plot(
        [-1, 9], [0, 0],
        linestyle="--", color="black", alpha=0.5,
        label="Hypotheses\nequally\nprobable"
        )
    axs[1].set_xlabel("Stimulus")
    plt.legend(loc="lower left")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.savefig(os.path.join(results_dir, "compete.png"))
        plt.savefig(os.path.join(results_dir, "compete.svg"))
    plt.close()

    if model_comparison_dir is not None:
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result_compete.npy"),
            {"fns": xlabels, "d": mean_log_odds},
            allow_pickle=True
        )
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots_compete.npy"),
            {"mean": mean_log_odds, "stderr": stderr_log_odds},
            allow_pickle=True
        )

    return mean_log_odds


def summarize(sound_group, expt_name, seeds, expts_to_analyze, inference_dir, results_dir=None, model_comparison_dir=None):

    inference_folder = os.path.join(inference_dir, expt_name, sound_group, "")
    results_dir = inference_folder if results_dir is None else results_dir

    # Van Noorden, 1975. Effects of frequency and rate.
    if "aba" in expts_to_analyze:
        delta_frequency = [1, 3, 6, 9, 12]
        delta_time = [67, 83, 100, 117, 134, 150]
        n_reps = 4
        M = np.full([len(delta_frequency), len(delta_time), 2, len(seeds)], np.nan)
        for fidx, df in enumerate(delta_frequency):
            for tidx, dt in enumerate(delta_time):
                for sidx, scene_type in enumerate(["galloping", "isochronous"]):
                    sound_fn = f"df{df}_dt{dt}_rep{n_reps}"
                    expt_folder = os.path.join(inference_folder, sound_fn, scene_type)
                    M[fidx, tidx, sidx, :] = read_json(expt_folder, seeds)
        plot_van_noorden(M, delta_frequency, delta_time, sound_group, expt_name, results_dir=results_dir, model_comparison_dir=model_comparison_dir)

    # Bouncing vs. crossing
    if "bouncing" in expts_to_analyze:
        M = np.full((3,), np.nan)
        bounce_hypotheses = ["crossing", "bouncing", "onesource"]
        for sidx, scene_type in enumerate(bounce_hypotheses):
            sound_fn = "Track17-01-rs-short"
            expt_folder = os.path.join(inference_folder, sound_fn, scene_type)
            result = np.median(read_json(expt_folder, seeds))
            M[sidx] = result

    # Cumulative repetition
    if "cumul" in expts_to_analyze:
        n_reps = range(1, 7)
        dts = [125]
        dfs = [4, 8]
        M = np.full((len(n_reps), len(dts), len(dfs), 2, len(seeds)), np.nan)
        for ridx, rep in enumerate(n_reps):
            for didx, dt in enumerate(dts):
                for dfidx, df in enumerate(dfs):
                    for sidx, scene_type in enumerate(["galloping", "isochronous"]):
                        sound_fn = f"df{df}_dt{dt}_rep{rep}"
                        expt_folder = os.path.join(inference_folder, sound_fn, scene_type)
                        result = read_json(expt_folder, seeds)
                        M[ridx, didx, dfidx, sidx, :] = result
        plot_cumulative(M, n_reps, sound_group, expt_name, results_dir=results_dir, model_comparison_dir=model_comparison_dir)

    # Captor
    if "captor" in expts_to_analyze:
        captor_conditions = ["none", "590", "1030", "1460"]
        M = np.full((len(captor_conditions), 3, len(seeds)), np.nan)
        for cidx, c in enumerate(captor_conditions):
            # C: Distractor + Captor, D: T + Distrator, 3S: each in a separate stream
            # 3S and C have the Ts separated from Distractor
            for sidx, scene_type in enumerate(["D", "C", "3S"]):
                sound_fn = f"compdown_f{c}"
                expt_folder = os.path.join(inference_folder, sound_fn, scene_type)
                M[cidx, sidx, :] = read_json(expt_folder, seeds)
        plot_captor(M, captor_conditions, sound_group, expt_name, results_dir=results_dir, model_comparison_dir=model_comparison_dir)

    # Competing explanations
    if "compete" in expts_to_analyze:
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
        print(compete_hypotheses)
        compete_hypotheses = [
            fn for fn in compete_hypotheses if fn not in ["0123", "0111", "1011", "0001", "1101"]
            ]
        frequencies = [
            [2800, 1556, 600, 333],
            [600, 333, 2800, 1556],
            [2800, 2642, 1556, 1468],
            [333, 314, 600, 566],
            [2800, 1556, 2642, 1468],
            [600, 333, 566, 314],
            [2800, 600, 1468, 314]
            ]
        conditions = ['isolate', 'isolate', 'isolate', 'isolate', 'absorb', 'absorb', 'absorb']
        sound_fns = []
        for i in range(len(conditions)):
            sound_fns.append(f"{conditions[i]}_A{frequencies[i][0]}_B{frequencies[i][1]}_D1-{frequencies[i][2]}_D2-{frequencies[i][3]}")
        
        M = np.full(
            (len(sound_fns), len(compete_hypotheses), len(seeds)),
            np.nan
            )
        for sfidx, sound_fn in enumerate(sound_fns):
            for sidx, scene_type in enumerate(compete_hypotheses):
                expt_folder = os.path.join(inference_folder, sound_fn, scene_type)
                M[sfidx, sidx, :] = read_json(expt_folder, seeds)
        plot_compete_average(M, compete_hypotheses, sound_fns, sound_group, expt_name, results_dir=results_dir, model_comparison_dir=model_comparison_dir)


def compare_two_aba_experiments(inference_folder_full, inference_folder_lesion, savepath, lim=15):
    """ Bistability: model lesion plot """

    # Both experiments should use the same settings
    seeds = range(10)
    delta_frequency = [1, 3, 6, 9, 12]
    delta_time = [67, 83, 100, 117, 134, 150]
    n_reps = 4

    log_odds = {}
    for key, inference_folder in {"full": inference_folder_full, "stationary": inference_folder_lesion}.items():
        M = np.full(
            [len(delta_frequency), len(delta_time), 2, len(seeds)], np.nan
            )
        for fidx, df in enumerate(delta_frequency):
            for tidx, dt in enumerate(delta_time):
                for sidx, scene_type in enumerate(["galloping", "isochronous"]):
                    sound_fn = "df{}_dt{}_rep{}".format(df, dt, n_reps)
                    expt_folder = os.path.join(inference_folder_full, sound_fn, scene_type)
                    M[fidx, tidx, sidx, :] = read_json(expt_folder, seeds)
        log_odds[key] = M[:, :, 1, :] - M[:, :, 0, :]
        log_odds[key] = np.nanmean(log_odds[key], axis=-1)

    t_h = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # milliseconds
    fusion_boundary = [2.9, 3, 3.1, 3, 4, 3.5, 4, 3, 3, 4]  # semitones
    temporal_coherence_boundary = [4, 4, 4.5, 4.8, 6, 7, 8, 9.5, 10.5, 12]  # semitones
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7.7, 3))
    for ax_idx, ax in enumerate(axs):
        if ax_idx == 0:
            p1, = ax.plot(
                t_h, fusion_boundary,
                "ko-", markerfacecolor="white", label="Two-sources"
                )
            p2, = ax.plot(
                t_h, temporal_coherence_boundary,
                "ko-", label="One-source"
                )
        else:
            ax.plot(
                t_h, fusion_boundary,
                "ko-", markerfacecolor="white", label="Two-sources"
                )
            ax.plot(
                t_h, temporal_coherence_boundary,
                "ko-", label="One-source"
                )

    # Experiment values
    DTs_full = []
    DFs_full = []
    DTs_stationary = []
    DFs_stationary = []
    DTs_all = []
    DFs_all = []
    # Colors
    CS_full = []
    CS_stationary = []
    c_full = np.array([75, 165, 219])/255
    c_stationary = np.array([53, 87, 166])/255
    for k in ["full", "stationary"]:
        for f_idx, df in enumerate(delta_frequency):
            for t_idx, dt in enumerate(delta_time):
                DTs_all.append(dt)
                DFs_all.append(df)
                if -lim <= log_odds[k][f_idx, t_idx] <= lim:
                    if k == "full":
                        DTs_full.append(dt)
                        DFs_full.append(df)
                        CS_full.append(c_full)
                    elif k == "stationary":
                        DTs_stationary.append(dt)
                        DFs_stationary.append(df)
                        CS_stationary.append(c_stationary)

    for ax in axs:
        ax.scatter(DTs_all, DFs_all, c=[np.zeros(3) for i in DTs_all], s=1)

    def plot_area(z1, z2, ax, area_color, label):
        pp = [(dt, df) for dt, df in zip(z1, z2)]
        cent = (sum([p[0] for p in pp])/len(pp), sum([p[1] for p in pp])/len(pp))
        pp.sort(key=lambda p: math.atan2(p[1]-cent[1], p[0]-cent[0]))
        ax.fill(
            [p[0] for p in pp], [p[1] for p in pp],
            area_color, facecolor=area_color, edgecolor=area_color,
            joinstyle="round", alpha=0.5, lw=15, label=label
            )

    # Actual
    plot_area(
        t_h + t_h, fusion_boundary + temporal_coherence_boundary,
        axs[0], "grey", "Human"
        )

    # Stationary
    plot_area(
        DTs_stationary, DFs_stationary,
        axs[2], c_stationary, "Stationary cov."
    )
    
    # Full
    plot_area(
        DTs_full, DFs_full,
        axs[1], c_full, "(Ours) Gen. model"
        )

    axs[0].set_ylim([0, 13.5])
    axs[0].set_xlim([55, 160])
    axs[0].set_ylabel("Frequency difference (semitones, $\Delta f$)")
    axs[0].set_xlabel("Onset-to-onset time (milliseconds, $\Delta t$)")

    # Create a legend for the first line.
    l1 = axs[0].legend(
        handles=[p1, p2], loc='upper left', title="Boundary", handletextpad=0.1
        )
    axs[0].add_artist(l1)

    plt.savefig(os.path.join(savepath, "aba_comparison.png"))
    plt.savefig(os.path.join(savepath, "aba_comparison.svg"))
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str, help="folder to save wavs")
    parser.add_argument("--expt-name", type=str, help="folder to group inference results together")
    parser.add_argument("--seeds", type=str, help="comma sep list of seeds")
    parser.add_argument("--inference-dir", type=str, default=os.environ["inference_dir"], help="top-level folder for inference")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--expts-to-analyze", type=str, default="aba,captor,cumul,compete", help="comma-separated list of expts included in this analysis")
    parser.add_argument("--model-comparison-dir", type=str, required=False, help="path to save enumerative results for model comparison")
    args = parser.parse_args()
    if args.results_dir is None:
        results_dir = os.path.join(os.environ["inference_dir"], args.expt_name, args.sound_group, "")
    else:
        results_dir = args.results_dir
    summarize(args.sound_group, args.expt_name, args.seeds.split(","), args.expts_to_analyze.split(","), args.inference_dir, results_dir=results_dir, model_comparison_dir=args.model_comparison_dir)
