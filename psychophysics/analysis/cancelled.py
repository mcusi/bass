import os
import json
from glob import glob
import numpy as np
import soundfile as sf
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import inference.io
import renderer.util
from util import softbound, manual_seed


def get_explanation_from_iterative(sound_name, sound_group, expt_name):
    """ Retrieve best explanations from inference in the generative model """

    # Get the folder where the inference runs are saved
    inference_dir = os.environ["inference_dir"]
    try:
        results_fn = os.path.join(inference_dir, expt_name, sound_group, sound_name, "results.json")
        with open(results_fn, "r") as f:
            results = json.load(f)
    except:
        return None

    # Look through all hypotheses
    round_idx = len(results["keep"])-1
    n_hypotheses = len(results["keep"][-1])+len(results["reject"][-1])
    for hypothesis_idx in range(n_hypotheses):
        if hypothesis_idx < 2:
            best_explanation = int(results["keep"][-1][hypothesis_idx])
        else:
            best_explanation = int(results["reject"][-1][hypothesis_idx-2])

        best_initialization = os.path.join(
            inference_dir, expt_name, sound_group, sound_name,
            f"round{round_idx:03d}-{best_explanation:03d}", ""
            )

        # Figure out which sources in a scene are a whistle source
        config = inference.io.get_config(best_initialization)
        scene, ckpt = inference.io.restore_hypothesis(best_initialization)
        source_types = config["hyperpriors"]["source_type"]["args"]
        whistle_idx = [
            i for i in range(len(source_types)) if source_types[i] == "whistle"
            ][0]
        n_whistle_sources = 0
        whistle_source_idxs = []
        for source_idx in range(scene.n_sources.item()):
            if scene.sources[source_idx].source_type_idx == whistle_idx:
                n_whistle_sources += 1
                whistle_source_idxs.append(source_idx)
        if n_whistle_sources > 0:
            break

    # No whistle
    if n_whistle_sources == 0:
        return None

    L = []
    for whistle_source_idx in whistle_source_idxs:
        mean_module = scene.sources[whistle_source_idx].gps["f0"].mean_module
        _loc = softbound(
            mean_module._mu_mu,
            mean_module.mu_hp[0],
            mean_module.mu_hp[1],
            (mean_module.mu_hp[1] - mean_module.mu_hp[0])/100
            )
        L.append(_loc.detach().numpy())

    return renderer.util.ERB_to_freq(np.mean(L))


def get_sound_from_iterative(inference_dir, sound_name, sound_group, expt_name):

    sequential_folder = os.path.join(
        os.environ["home_dir"], "comparisons", "results",
        "gen-model", sound_group, expt_name, ""
        )
    os.makedirs(sequential_folder, exist_ok=True)
    inference_dir = os.environ["inference_dir"]
    try:
        results_fn = os.path.join(inference_dir, expt_name, sound_group, sound_name,"results.json")
        with open(results_fn, "r") as f:
            results = json.load(f)
    except:
        return None

    round_idx = len(results["keep"])-1
    hypothesis_idx = 0
    best_explanation = int(results["keep"][-1][hypothesis_idx])

    best_initialization = os.path.join(
        inference_dir, expt_name, sound_group, sound_name,
        f"round{round_idx:03d}-{best_explanation:03d}", ""
        )

    _, ckpt = inference.io.restore_hypothesis(best_initialization)
    scene = ckpt["metrics"].best_scene

    batch_idx = ckpt["metrics"].best_scene_scores.argmax()
    for source_idx, source in enumerate(scene.sources):
        sf.write(
            os.path.join(sequential_folder, f"{sound_name}_round{round_idx:03d}-{best_explanation:03d}_source{source_idx:02d}_estimate.wav"),
            source.source_wave.detach().numpy()[batch_idx, :],
            scene.audio_sr
            )

    return None


def get_f0_and_harmonic_idx(sound_name):
    # Example sound name: cancelled_f0400_n08_durf0750_durt0100_nt05
    f0 = int(sound_name.split("_")[1][1:])
    harmonic_idx  = int(sound_name.split("_")[2][1:])
    return f0, harmonic_idx


def compute_error_in_percent(sound_name, estimated_f0):
    # Example sound name: cancelled_f0400_n08_durf0750_durt0100_nt05
    # Note that harmonic_idx=1 in the paper and here is the fundamental
    f0 = int(sound_name.split("_")[1][1:])
    harmonic_idx  = int(sound_name.split("_")[2][1:])
    return harmonic_idx, (estimated_f0 - f0*harmonic_idx)/(1.*f0*harmonic_idx)


def psychophysics(sound_group, expt_name, results_dir=None, model_comparison_dir=None):

    inference_dir = os.environ["inference_dir"]
    savepath = os.path.join(inference_dir, expt_name, sound_group, "")

    n_harmonics_to_test = np.asarray([1, 2, 3, 10, 11, 12, 18, 19, 20])
    sound_names = glob(os.path.join(savepath, "cancel*", ""))
    sound_names = [[s for s in sn.split(os.sep) if len(s) > 0][-1] for sn in sound_names]
    f0s = np.unique([
        int(sound_name.split("_")[1][1:]) for sound_name in sound_names
        ])
    results_per_f0 = {}
    results_per_idx = {}
    for f0 in f0s:
        results_per_f0[f0] = {
            "harmonic_idxs": [],
            "percent_errors": [],
            "whistle_present": []
            }
        sound_names_f0 = [sn for sn in sound_names if f"_f{f0:04d}_" in sn]
        for sound_name in sound_names_f0:
            
            f0_estimate = get_explanation_from_iterative(sound_name, sound_group, expt_name)
            print(f0_estimate)
            if f0_estimate is None:  # no whistle source was found
                harmonic_idx, error = compute_error_in_percent(sound_name, 0)
                results_per_f0[f0]["harmonic_idxs"].append(harmonic_idx)
                percent_error = np.nan
                whistle_present = False
            else:
                harmonic_idx, error = compute_error_in_percent(sound_name, f0_estimate)
                results_per_f0[f0]["harmonic_idxs"].append(harmonic_idx)
                percent_error = 100*error
                whistle_present = True
            results_per_f0[f0]["percent_errors"].append(percent_error)
            results_per_f0[f0]["whistle_present"].append(True)
            
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

    print(results_per_f0)
    plot_bars(
        results_per_f0,
        sound_group,
        expt_name,
        savepath if results_dir is None else results_dir
        )

    if model_comparison_dir is not None:
        np.save(os.path.join(
            model_comparison_dir, sound_group, expt_name, "for_plots.npy"
            ), results_per_f0
            )


def plot_bars(D, sound_group, network, savepath):

    # Model Data
    Xs = [1, 2, 3, 10, 11, 12, 18, 19, 20]
    errors_model = np.array([np.abs(v["percent_errors"] + [np.nan for _ in range(len(Xs)-len(v["percent_errors"]))]) for k, v in D.items()])

    # People Data
    b = [(1, -1.8), (1, -0.7), (1, -0.4), (1, 0.1), (1, 2.4), (2, -1.0), (2, -0.1), (2, 0.4), (2, 0.4), (2, 0.7), (3, -3.3), (3, -1.3), (3, -0.1), (3, 1.0), (3, 2.4), (4, -1.6), (4, -0.7), (4, -0.1), (4, 0.7), (4, 1.0), (5, -1.3), (5, 0.4), (5, 1.0), (5, 1.3), (5, 1.8), (6, -1.3), (6, -0.1), (6, 0.4), (6, 0.7), (6, 0.7), (7, -2.7), (7, -2.1), (7, -1.3), (7, -0.7), (7, 0.4), (8, -1.6), (8, -0.4), (8, 0.1), (8, 0.4), (8, 1.3), (9, -1.3), (9, -1.0), (9, 0.1), (9, 0.7), (9, 1.8), (10, -2.1), (10, -1.8), (10, -1.0), (10, -0.7), (10, -0.1), (11, -3.3), (11, -2.1), (11, -1.3), (11, -1.0), (11, np.nan), (12, 0.4), (12, 0.7), (12, 1.3), (12, 2.4), (12, 3.3), (13, -5.2), (13, -4.4), (13, 0.1), (13, 2.7), (13, np.nan), (14, -7.8), (14, -2.7), (14, -2.7), (14, -2.4), (14, -0.1), (15, -5.5), (15, -4.1), (15, -3.3), (15, 3.3), (15, 9.8), (16, -3.3), (16, -3.3), (16, -1.8), (16, 1.3), (16, 1.3), (17, -18.9), (17, -8.4), (17, -3.5), (17, 2.7), (17, np.nan), (18, -13.5), (18, -8.7), (18, -8.1), (18, -7.5), (18, -7.2), (19, -17.2), (19, -3.3), (19, -2.7), (19, 6.1), (19, 7.5), (20, -15.2), (20, -7.5), (20, -7.5), (20, 1.3), (20, 5.0)]
    m = [(1, -1.1), (1, 0.8), (1, 2.0), (1, 2.8), (1, 20.0), (2, -0.6), (2, 0.0), (2, 0.6), (2, 0.6), (2, 1.1), (3, -1.1), (3, 0.0), (3, 0.8), (3, 1.4), (3, 2.3), (4, -1.7), (4, -1.7), (4, 0.3), (4, 0.3), (4, 2.0), (5, -1.1), (5, 0.3), (5, 0.3), (5, 1.4), (5, 1.7), (6, -2.0), (6, -1.4), (6, 0.8), (6, 2.8), (6, 6.2), (7, -1.7), (7, -0.3), (7, 1.1), (7, 2.0), (7, 2.3), (8, -0.3), (8, 0.3), (8, 0.8), (8, 3.1), (8, 3.1), (9, -3.4), (9, -1.1), (9, -0.8), (9, -0.3), (9, 3.1), (10, -2.3), (10, 0.0), (10, 1.1), (10, 2.0), (10, 5.4), (11, -3.7), (11, -3.7), (11, -1.1), (11, -0.6), (11, 0.8), (12, -3.1), (12, -2.8), (12, -2.3), (12, -2.0), (12, -0.8), (13, -3.4), (13, -2.8), (13, -1.7), (13, -1.4), (13, 0.3), (14, -9.9), (14, -3.4), (14, -1.7), (14, 5.4), (14, 15.5), (15, -4.8), (15, -2.5), (15, -0.8), (15, 2.5), (15, 5.6), (16, np.nan), (16, np.nan), (16, np.nan), (16, np.nan), (16, np.nan), (17, np.nan), (17, np.nan), (17, np.nan), (17, np.nan), (17, np.nan), (18, np.nan), (18, np.nan), (18, np.nan), (18, np.nan), (18, np.nan), (19, np.nan), (19, np.nan), (19, np.nan), (19, np.nan), (19, np.nan), (20, np.nan), (20, np.nan), (20, np.nan), (20, np.nan), (20, np.nan)]
    w = [(1, -1.8), (1, -0.4), (1, 0.1), (1, 0.7), (1, 0.7), (2, -0.4), (2, 0.4), (2, 0.7), (2, 0.7), (2, 0.7), (3, -0.4), (3, 1.0), (3, 1.8), (3, 2.1), (3, 3.5), (4, 0.1), (4, 0.7), (4, 1.6), (4, 1.8), (4, 2.4), (5, 0.4), (5, 1.3), (5, 1.6), (5, 2.4), (5, 2.7), (6, 0.1), (6, 1.0), (6, 1.6), (6, 2.1), (6, 3.0), (7, 1.3), (7, 2.4), (7, 2.7), (7, 3.3), (7, 3.8), (8, -1.8), (8, -1.3), (8, -0.4), (8, -0.1), (8, 0.1), (9, -0.1), (9, 0.7), (9, 1.0), (9, 1.6), (9, 1.8), (10, 0.1), (10, 0.4), (10, 0.7), (10, 0.7), (10, 0.7), (11, -0.7), (11, 1.6), (11, 2.1), (11, 2.1), (11, 3.0), (12, 0.4), (12, 1.6), (12, 1.8), (12, 2.4), (12, 3.0), (13, -0.7), (13, -0.1), (13, 0.4), (13, 0.7), (13, 1.3), (14, -0.7), (14, -0.1), (14, 0.4), (14, 1.3), (14, 2.7), (15, -2.1), (15, -1.0), (15, 1.3), (15, 4.1), (15, 4.4), (16, -3.8), (16, -0.4), (16, 1.0), (16, 3.0), (16, 3.8), (17, -3.5), (17, -1.6), (17, 1.8), (17, 4.1), (17, np.nan), (18, -8.7), (18, -8.1), (18, 7.2), (18, 9.5), (18, np.nan), (19, -8.4), (19, -0.1), (19, np.nan), (19, np.nan), (19, np.nan), (20, -18.6), (20, -16.9), (20, -13.5), (20, 7.8), (20, 17.2)]
    errors_human = np.full([3, 5, len(Xs)], np.nan)
    for participant_idx, participant in enumerate([b, m, w]):
        for x_idx, x in enumerate(Xs):
            percent_error = [p[1] for p in participant if p[0] == x]
            for trial_idx, err in enumerate(percent_error):
                errors_human[participant_idx, trial_idx, x_idx] = np.abs(err)

    errors_human = errors_human.reshape(15, len(Xs))

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5.5, 2))
    for dataset_idx, abs_av_percent_errors in enumerate([errors_human, errors_model]):

        error_bin = {}
        lt_thresholds = [2, 10]
        prev_error_thresh = 0
        for thresh_idx, thresh in enumerate(lt_thresholds):
            if thresh_idx == 0:
                error_bin[thresh] = (abs_av_percent_errors <= thresh).sum(0)/abs_av_percent_errors.shape[0]
            else:
                error_bin[thresh] = ((prev_error_thresh < abs_av_percent_errors)*(abs_av_percent_errors <= thresh)).sum(0)/abs_av_percent_errors.shape[0]
            prev_error_thresh = thresh

        error_bin[">10"] = (prev_error_thresh < abs_av_percent_errors).sum(0)/abs_av_percent_errors.shape[0]
        error_bin["No match"] = np.isnan(abs_av_percent_errors).sum(0)/abs_av_percent_errors.shape[0]

        all_bins = lt_thresholds + [">10", "No match"]
        x = 0.5*np.arange(3)
        width = 0.1
        bottom = 0
        colors = [
            [50, 58, 168], [98, 105, 208], [137, 143, 220], [192, 176, 232]
            ]
        colors = [np.array(c)/255 for c in colors]
        ax = axs[dataset_idx]
        base_idxs = np.array([0, 3, 6])
        for b_idx, b in enumerate(all_bins):
            for data_idx, mult in enumerate([-1,0,1]):
                Y = np.array([error_bin[b][i] for i in base_idxs+data_idx])
                if b_idx == 0:
                    B = np.zeros(base_idxs.shape)
                else:
                    B = np.array([bottom[i] for i in base_idxs+data_idx])
                labrador = (b if isinstance(b, str) else "<"+str(b)) if data_idx == 0 else "__nolabel__"
                ax.bar(
                    x + mult*width, Y, width,
                    bottom=B,  color=colors[b_idx],
                    edgecolor="black", label=labrador
                    )
            bottom = bottom + error_bin[b]

    axs[0].set_xlabel("Harmonic number of gated component")
    axs[0].set_ylabel("Proportion of trials")
    axs[0].set_xticks([-width, 0, width, 0.5-width, 0.5, 0.5+width, 1-width, 1, 1+width]); 
    axs[1].set_xticks([-width, 0, width, 0.5-width, 0.5, 0.5+width, 1-width, 1, 1+width]); 
    axs[0].set_xticklabels(["1", "2\nLow", "3", "10", "11\nMid", "12", "18", "19\nHigh", "20"])
    axs[1].set_xticklabels(["1", "2\nLow", "3", "10", "11\nMid", "12", "18", "19\nHigh", "20"])
    axs[0].set_title("Hartmann and Goupell, 2006")
    axs[1].set_title(f"Gen model: {sound_group} {network}", fontsize="small")
    plt.tight_layout()
    print(os.path.join(savepath, "stacked.png"))
    plt.savefig(os.path.join(savepath, "stacked.png"))
    plt.savefig(os.path.join(savepath, "stacked.svg"))
    plt.close()


if __name__ == "__main__":
    manual_seed(0)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-group", type=str, help="folder to save wavs")
    parser.add_argument("--expt-name", type=str, help="folder to group inference results together")
    parser.add_argument("--inference-dir", type=str, default=os.environ["inference_dir"], help="top-level folder for inference")
    parser.add_argument("--sound-dir", type=str, default=os.environ["sound_dir"], help="top-level folder for inference")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--model-comparison-dir", type=str, required=False)
    args = parser.parse_args()
    os.environ["inference_dir"] = args.inference_dir
    os.environ["sound_dir"] = args.sound_dir
    psychophysics(args.sound_group, args.expt_name, results_dir=args.results_dir, model_comparison_dir=args.model_comparison_dir)
