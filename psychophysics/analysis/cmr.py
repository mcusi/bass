import os
import json
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf

import inference.io
import psychophysics.analysis.thresholds as pat
from util import manual_seed


def get_sound_from_iterative(inference_dir, blank, sound_group, expt_name, tone_levels, bws=[100, 200, 400, 1000]):

    exptpath = os.path.join(inference_dir, expt_name, sound_group, "")
    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons", "results", "gen-model", sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)

    conditions = ["mult", "rand"]
    st_results = {}
    for condition in conditions:
        st_results[condition] = {}
        for bw in bws:
            for tone_level in tone_levels:
                for seed in range(10):
                    sound_name = f"{condition}_tone_bw{bw}_toneLevel{tone_level}_seed{seed}-0"
                    folder = exptpath + sound_name + "/"

                    with open(folder + "results.json", "r") as f:
                        results = json.load(f)

                    round_idx = len(results["keep"]) - 1
                    expl = int(results["keep"][-1][0])

                    best_initialization = os.path.join(
                        inference_dir, expt_name, sound_group, sound_name, f"round{round_idx:03d}-{expl:03d}", ""
                        )
                    _, ckpt = inference.io.restore_hypothesis(best_initialization)
                    scene = ckpt["metrics"].best_scene

                    # Save the sounds
                    batch_idx = ckpt["metrics"].best_scene_scores.argmax()
                    for source_idx, source in enumerate(scene.sources):
                        sf.write(
                            os.path.join(sequential_folder, f"{sound_name}_round{round_idx:03d}-{expl:03d}_source{source_idx:02d}_estimate.wav"),
                            source.source_wave.detach().numpy()[batch_idx, :],
                            scene.audio_sr
                        )

    return


def full_analysis(sound_group, expt_name, seeds, results_dir):

    savepath = os.path.join(os.environ["inference_dir"], expt_name, sound_group, "")
    if results_dir is None:
        results_dir = savepath

    collected_results = []
    for seed in seeds:
        collected_results_seed, for_plots = compile_results(sound_group, expt_name, seed)
        collected_results.append(collected_results_seed)

    compiled_results = {}
    for bandwidth in for_plots["bandwidths"]:
        compiled_results[bandwidth] = pd.concat([cr[bandwidth] for cr in collected_results])

    one_interval_threshold(
        compiled_results, for_plots["bandwidths"],
        for_plots, results_dir,
        sound_group, expt_name,
        model_comparison_dir=model_comparison_dir
        )

    return


def compile_results(sound_group, expt_name, seed):

    settings = np.load(
        os.path.join(os.environ["sound_dir"], sound_group, f"mr_expt1_settings_seed{seed}.npy"),
        allow_pickle=True
        ).item()

    exemplars = settings["n_trials"]
    sound_types = ["rand", "mult"]
    signal_levels_in_stimulus = settings["tone_levels"]
    signal_levels_in_stimulus = np.concatenate([[0], signal_levels_in_stimulus])
    explanations = sorted(["noise-only", "noise+tone"])
    bandwidths = settings["bandwidths"]
    bandwidths = sorted(bandwidths)

    table_columns = ["seed", "exemplar", "sound_type", "signal_level"]
    table_columns.extend(explanations)
    compiled_results = {}
    for bandwidth in bandwidths:  # stimulus param
        T = []
        for exemplar in range(exemplars):  # stimulus param
            for sound_type in sound_types:  # stimulus param
                for signal_level in signal_levels_in_stimulus:  # stimulus param
                    if signal_level == 0:
                        sound_name = f'{sound_type}_noTone_bw{bandwidth}_seed{seed}-{exemplar}'
                    else:
                        sound_name = f'{sound_type}_tone_bw{bandwidth}_toneLevel{signal_level}_seed{seed}-{exemplar}'
                    elbo_vals = {}
                    sound_folder = os.path.join(os.environ["inference_dir"], expt_name, sound_group, sound_name, "")
                    for explanation in explanations:  # init param
                        initializations_of_this_explanation = glob(os.path.join(sound_folder, f"*{explanation}*", ""))
                        elbos = []
                        for initialization in initializations_of_this_explanation:
                            try:
                                with open(initialization + "metrics.json", "r") as f:
                                    metrics = json.load(f)
                                    elbos.append(metrics["elbo"])
                            except:
                                print("missing ", initialization)
                        if len(elbos) > 0:
                            elbo_vals[explanation] = np.max(elbos)
                        else:
                            elbo_vals[explanation] = np.nan
               
                    T.append((seed, exemplar, sound_type, signal_level, *[elbo_vals[e] for e in explanations] ))

        D = pd.DataFrame.from_records(T, columns=table_columns)
        compiled_results[bandwidth] = D

    for_plots = {
        "sound_types": sound_types,
        "signal_levels": signal_levels_in_stimulus,
        "explanations": explanations,
        "bandwidths": bandwidths
        }
    return compiled_results, for_plots


def one_interval_threshold(compiled_results, bandwidths, for_plots, savepath, sound_group, expt_name, explanation1="noise+tone", explanation2="noise-only", model_comparison_dir=None):

    sound_types = for_plots["sound_types"]
    signal_levels = for_plots["signal_levels"]

    # One interval plot
    interval1 = {}
    for bandwidth in bandwidths:
        D = compiled_results[bandwidth]
        interval1[bandwidth] = {}
        for sound_type in sound_types:
            interval1[bandwidth][sound_type] = []
            for signal_level in signal_levels:
                b = (D["sound_type"] == sound_type) & (D["signal_level"] == signal_level)
                interval1[bandwidth][sound_type].append(D[b][explanation1]-D[b][explanation2])
            interval1[bandwidth][sound_type] = np.stack(interval1[bandwidth][sound_type])

    results = {}
    # interval1[bandwidth][condition] = array(n_levels, n_seeds)
    for bandwidth in bandwidths:
        for sound_type in sound_types:
            x = signal_levels[1:]
            odds = interval1[bandwidth][sound_type]
            boundaries = pat.average_crossing_points_log_axis(x, odds[1:, :], 0)
            results[(bandwidth, sound_type)] = boundaries

    # Save log odds for use with neural networks
    comparison_results = {}
    for sound_type in sound_types:
        comparison_results[sound_type] = {}
        for bandwidth in bandwidths:
            comparison_results[sound_type][bandwidth] = {}
            level_by_seed_arr = interval1[bandwidth][sound_type]
            for signal_idx, signal_level in enumerate(signal_levels[1:]):
                comparison_results[sound_type][bandwidth][signal_level] = level_by_seed_arr[signal_idx,:].tolist()

    if model_comparison_dir is not None:
        np.save(
            os.path.join(model_comparison_dir, "log_odds.npy"),
            comparison_results, allow_pickle=True
            )

    # Make experiment plot
    r = np.stack([results[(bw, "rand")] for bw in bandwidths])
    m = np.stack([results[(bw, "mult")] for bw in bandwidths])
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(5.8, 1.4))
    R_means = [53.7, 55.1, 57.5, 58.7, 60.8, 59.8]
    M_means = [53, 56, 56.2, 55.6, 51.3, 49.8]
    R_standard_deviations = np.array([1.25, 1.25, 1.5, 1.0, 0.75, 1.])/np.sqrt(5)  # 5 participants
    M_standard_deviations = np.array([1.25, 1.5, 1.25, 1.3, 1., 1.25])/np.sqrt(5)
    axs[0].errorbar(
        [25, 50, 100, 200, 400, 1000], R_means,
        R_standard_deviations,
        fmt="ko-", markerfacecolor="white",
        label="Random noise"
        )
    axs[0].errorbar(
        [25, 50, 100, 200, 400, 1000], M_means,
        M_standard_deviations,
        fmt="ko-", markeredgecolor=(0, 0, 0, 1.0),
        markerfacecolor=(0, 0, 0, 0.5), alpha=0.7,
        label="Co-modulated noise"
        )
    axs[0].set_title("Hall et al. (1984)")
    axs[0].text(
        0.01, 0.82, 'fm=50Hz\n2-interval',
        size=8, color='black', transform=axs[0].transAxes
        )

    model_r_mean = np.mean(r, axis=1)
    model_m_mean = np.mean(m, axis=1)
    model_r_std = np.std(r, axis=1)/np.sqrt(r.shape[1])
    model_m_std = np.std(m, axis=1)/np.sqrt(m.shape[1])

    model_dict_to_save = {"rand": model_r_mean, "mult": model_m_mean}
    if model_comparison_dir is not None:
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "result.npy"),
            model_dict_to_save, allow_pickle=True
            )
        for_plots = {"r_mean": model_r_mean, "m_mean": model_m_mean, "r_std": model_r_std, "m_std": model_m_std}
        np.save(
            os.path.join(model_comparison_dir, sound_group, expt_name, "for_plots.npy"),
            for_plots, allow_pickle=True
            )

    axs[1].errorbar(
        bandwidths, model_r_mean, model_r_std,
        fmt="ko-", markerfacecolor="white",
        label="Random noise"
        )
    axs[1].errorbar(
        bandwidths, model_m_mean, model_m_std,
        fmt="ko-", markeredgecolor=(0, 0, 0, 1.0),
        markerfacecolor=(0, 0, 0, 0.5), alpha=0.7,
        label="Co-modulated noise"
        )
    axs[1].text(
        0.01, 0.82, 'fm=10Hz\n1-interval',
        size=8, color='black', transform=axs[1].transAxes
        )
    axs[0].set_xlabel("Bandwidth (Hz)")
    axs[1].set_xlabel("Bandwidth (Hz)")
    axs[0].set_ylabel("Threshold (dB)")
    axs[1].set_ylim([40, 80])
    axs[0].set_ylim([40, 80])
    axs[1].set_title(" ".join(["Gen model:", sound_group, expt_name]), fontsize="x-small")
    axs[0].legend(fontsize="small")
    print(os.path.join(savepath, "cmr_1interval_thresholds.png"))
    print(os.path.join(savepath, "cmr_1interval_thresholds.svg"))
    plt.savefig(os.path.join(savepath, "cmr_1interval_thresholds.png"))
    plt.savefig(os.path.join(savepath, "cmr_1interval_thresholds.svg"))
    plt.close()

    return


if __name__ == "__main__":
    manual_seed(0)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str,
                        help="folder where wavs are saved")
    parser.add_argument("--expt-name", type=str,
                        help="folder with grouped inference results")
    parser.add_argument("--inference-dir", type=str, default=os.environ["inference_dir"],
                        help="top-level folder for inference")
    parser.add_argument("--sound-dir", type=str, default=os.environ["sound_dir"],
                        help="top-level folder for sounds")
    parser.add_argument("--results-dir", type=str, default="")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--model-comparison-dir", type=str, required=False)
    args = parser.parse_args()
    seeds = args.seeds.split(",")
    os.environ["inference_dir"] = args.inference_dir
    os.environ["sound_dir"] = args.sound_dir
    if len(args.results_dir) == 0:
        results_dir = os.path.join(
            args.inference_dir, args.expt_name, args.sound_group, ""
            )
    else:
        results_dir = args.results_dir
    full_analysis(
        args.sound_group,
        args.expt_name,
        seeds,
        results_dir,
        model_comparison_dir=args.model_comparison_dir
        )
