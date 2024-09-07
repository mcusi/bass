import os
import yaml
import argparse
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
import pingouin as pg


def main(csv_name, expt_name, sound_group, expt_idx):
    """ Analyzes csv returned from Experiment 2 """

    # Load in experiment config and raw results
    mixture_info = np.load(os.path.join(
        os.environ["sound_dir"],
        sound_group,
        "real_label_dict.npy"
        ), allow_pickle=True).item()
    raw_results = pd.read_csv(os.path.join(
        os.environ["home_dir"],
        "audio",
        "results",
        csv_name + ".csv"
        ), dtype=object)
    full_experiment_name = "-".join([
        expt_name, sound_group, f"{expt_idx:03d}", "expt2"
        ])
    experiment_folder = os.path.join(
        os.environ["sound_dir"], sound_group,
        "experiments", full_experiment_name, ""
        )
    with open(os.path.join(experiment_folder, "config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trial_info = pd.read_pickle(os.path.join(
        experiment_folder, "all_trial_info.pkl"
        ))
    savepath = os.path.join(experiment_folder, csv_name, "")
    os.makedirs(savepath, exist_ok=True)

    # Loop through participants to process results and apply inclusion criteria
    trial_table = []
    table_columns = [
        "worker_idx", "sound_number", "exemplar",
        "response_matrix", "col_source_idxs",
        "row_source_idxs", "shuffled_response_matrix"
        ]
    n_workers = len(raw_results["WorkerId"])
    workers_to_keep = []
    workers_to_skip = []
    headphone_check_failed = []
    for worker_idx in range(n_workers):

        # Exclude participants if they failed the headphone check
        # or if they did not pass the practice trials
        if int(raw_results["Answer.HC_correct"][worker_idx]) < 5 or int(raw_results["Answer.screen_correct"][worker_idx]) < 2:
            workers_to_skip.append(worker_idx)
            headphone_check_failed.append(worker_idx)
            continue
        # Basic info to reconstruct what happened
        column_shuffles = raw_results["Answer.col_shuffle"][worker_idx].split("|")
        row_shuffles = raw_results["Answer.row_shuffle"][worker_idx].split("|")
        soundnums_played = [int(sn) for sn in raw_results["Answer.soundnum_for_trial"][worker_idx].split("|")]

        # Set up catch trial checks
        worker_table = []
        if config["n_catch_trials"] > 0:
            # Catch trials will always be the first (N=n_catch_trials)
            # sound numbers, although the order in the trials will be
            # randomized
            catch_trial_results = []

            def add_to_worker_table(sound_number):
                return sound_number >= config["n_catch_trials"] + config.get("n_overcombine_catch_trials", 0)

        else:
            def add_to_worker_table(_):
                return True

        for trial_idx, soundnum_played in enumerate(soundnums_played):

            # Sounds played
            sound_info = trial_info.loc[trial_info.sound_number == soundnum_played]
            exemplar = sound_info["exemplar"].item()
            # Get shuffle info in order to unshuffle
            row_shuffle = [int(r_idx) for r_idx in row_shuffles[trial_idx].split("_")]
            column_shuffle = [int(c_idx) for c_idx in column_shuffles[trial_idx].split("_")]
            # actual_response: (row_idx, col_idx)
            actual_response = [(int(numstr[0]), int(numstr[1])) for numstr in raw_results[f"Answer.resp_for_trial_{trial_idx}"][worker_idx].split("_")]

            # First, recreate the checkmark matrix that
            # the participant was filling in
            n_rows = len(row_shuffle)
            n_cols = len(column_shuffle)
            checkmarks = np.zeros((n_rows, n_cols))
            for row_idx, col_idx in actual_response:
                checkmarks[row_idx, col_idx] = 1
            # Second, "unshuffle" the matrix so that all
            # participant answers will be in the same order
            # That ordering should be the ordering of the
            # sound_group/single_sources/FUSS-train_XXXXX_sourceXX.wav
            # First do the rows
            unshuffled_rows = [np.zeros(n_cols) for _ in range(n_rows)]
            for idx_in_participant_grid, actual_idx in enumerate(row_shuffle):
                unshuffled_rows[actual_idx] = checkmarks[idx_in_participant_grid, :]
            checkmarks_with_unshuffled_rows = np.stack(
                unshuffled_rows, axis=0
                )
            # Then do the cols as well
            unshuffled_cols_too = [np.zeros(n_rows) for _ in range(n_cols)]
            for idx_in_participant_grid, actual_idx in enumerate(column_shuffle):
                unshuffled_cols_too[actual_idx] = checkmarks_with_unshuffled_rows[:, idx_in_participant_grid]
            response_matrix = np.stack(unshuffled_cols_too, axis=1)

            if add_to_worker_table(soundnum_played):
                # Save the unshuffled response matrix for later analysis
                worker_table.append([
                    worker_idx,
                    soundnum_played,
                    exemplar,
                    response_matrix,
                    sound_info["rec_source_idxs"].item(),
                    sound_info["model_row_idxs"].item(),
                    checkmarks
                    ])
            else:
                if response_matrix.shape[0] == response_matrix.shape[1]:
                    # Standard catch trials should have a diagonal matrix
                    correct_answer = np.eye(response_matrix.shape[0])
                    ct_type = "diagonal"
                elif response_matrix.shape[0] != response_matrix.shape[1]:
                    # This is an overcombine trial - these are always
                    # encoded with the first two sounds combined
                    # Generated these for experiment but ignored them
                    # for inclusion criteria
                    correct_answer = np.zeros((n_rows, n_cols))
                    assert n_rows == 2, "Overcombine catch trial should \
                                            have two rows"
                    assert n_cols == 3, "Overcombine catch trial should \
                                            have three columns"
                    correct_answer[0, 0] = 1
                    correct_answer[0, 1] = 1
                    correct_answer[1, 2] = 1
                    ct_type = "oc"
                worker_responded_correctly = np.all(
                    correct_answer == response_matrix
                    )
                catch_trial_results.append((
                    worker_responded_correctly, ct_type
                    ))

        # Apply catch trial inclusion criteria
        if config["n_catch_trials"] > 0:
            if len([_ for _ in catch_trial_results if _[1] == "oc"]) > 0:
                diagonal_result = np.mean([_[0] for _ in catch_trial_results if _[1] == "diagonal"])
                # oc_result = np.mean([_[0] for _ in catch_trial_results if _[1] == "oc"])
                # Ignore overcombine catch trials
                skip_worker = (diagonal_result < 0.7)
                print(f"Worker {worker_idx}: {diagonal_result < 0.7}")
            else:
                performance = np.mean([ctr[0] for ctr in catch_trial_results])
                skip_worker = performance < 0.7
            if skip_worker:
                workers_to_skip.append(worker_idx)
            else:
                workers_to_keep.append(worker_idx)
                trial_table.extend(worker_table)
        else:
            trial_table.extend(worker_table)

    print("N workers kept: ", len(workers_to_keep))
    print("Workers to exclude: ", workers_to_skip)
    print("N failed headphone check: ", len(headphone_check_failed))
    print("Total workers to exclude: ", len(workers_to_skip))
    get_demographics(raw_results, workers_to_keep)
    experiment_results = pd.DataFrame(trial_table, columns=table_columns)
    experiment_results.to_pickle(os.path.join(
        experiment_folder, "results.pkl"
        ))

    workers_to_keep = [
        w_idx for w_idx in range(n_workers) if w_idx not in workers_to_skip
        ]
    interclasscorr(experiment_results, workers_to_keep)
    # Random baseline, if workers are just guessing
    summary_stats_guess, sim_stats_by_worker = random_simulations(
        experiment_results, workers_to_keep, trial_info, 100
    )
    # Count up different kinds of perceptual organization
    # deviations from single sources
    summary_stats_per_worker = participant_deviations(
        experiment_results,
        workers_to_keep,
        savepath,
        mixture_info=mixture_info,
        trial_info=trial_info
        )
    # Compare guessing and participants to get a summary plot
    summary_plot(
        sim_stats_by_worker,
        summary_stats_per_worker,
        len(workers_to_keep),
        full_experiment_name,
        savepath,
        permutations_for_quantiles=summary_stats_guess
        )


def get_demographics(raw_results, workers_to_keep):
    """ Collect demographics of age and gender
        for workers included in the experiment
    """
    genders = {
        "female": 0,
        "male": 0,
        "non-binary": 0,
        "other": 0
        }
    ages = []
    for worker_idx in workers_to_keep:
        gender = raw_results.iloc[worker_idx]["Answer.Q1Gender"]
        age = raw_results.iloc[worker_idx]["Answer.Q2age"]
        try:
            ages.append(int(age))
        except:
            print(f"Age {age} broke")
        try:
            if gender[0].lower() == "f":
                genders["female"] += 1
            elif gender[0].lower() == "m":
                genders["male"] += 1
            elif "non" in gender:
                genders["non-binary"] += 1
        except:
            print("Other gender", gender)
            genders["other"] += 1
    print("Gender: ", genders)
    print("Age mean: ", np.mean(ages))
    print("Age std: ", np.std(ages))


def interclasscorr(experiment_results, workers_to_keep):
    """ Determines the inter-class correlation over
        each inferred source / premixture sound pair
    """
    scene_idxs = experiment_results.exemplar.unique()
    table_columns = ["targets", "raters", "ratings"]
    T = []
    for scene_idx in scene_idxs:
        for worker_idx in workers_to_keep:
            response_matrix = experiment_results[
                (experiment_results.worker_idx == worker_idx) &
                (experiment_results.exemplar == scene_idx)
                ]["response_matrix"].item()
            n_rows, n_columns = response_matrix.shape
            for row_idx in range(n_rows):
                for col_idx in range(n_columns):
                    target = f"{scene_idx}_{row_idx}_{col_idx}"
                    rater = worker_idx
                    rating = response_matrix[row_idx, col_idx]
                    T.append((target, rater, rating))
    df = pd.DataFrame(T, columns=table_columns)
    icc = pg.intraclass_corr(
        data=df,
        targets='targets',
        raters='raters',
        ratings='ratings'
        )
    print(icc.set_index('Type'))


def summary_plot(guessing, experiment, n_workers, full_experiment_name, experiment_folder, permutations_for_quantiles=None):
    """ Figure 10G: get summary statistics and statistical
        significance as compared to permutations of the data
    """

    # Collect mean and standard deviations for experiment conditions
    ks = [
        "absent",
        "oversegment",
        "overcombine",
        "no_deviations",
        "no_deviations_and_most_common"
        ]
    summary_matrix = np.full((len(ks), n_workers), np.nan)
    guessing_matrix = np.full((len(ks), n_workers), np.nan)
    for i, worker_idx in enumerate(guessing.keys()):
        worker_g = guessing[worker_idx]
        worker_e = experiment[worker_idx]
        for k_idx, k in enumerate(ks):
            summary_matrix[k_idx, i] = worker_e[k]
            guessing_matrix[k_idx, i] = worker_g[k]
    # Actual data
    s_means = np.nanmean(summary_matrix, axis=1)
    s_stds = np.nanstd(summary_matrix, axis=1)/np.sqrt(n_workers)
    # Permuted data
    g_means = np.nanmean(guessing_matrix, axis=1)
    g_stds = np.nanstd(guessing_matrix, axis=1)/np.sqrt(n_workers)
    print("Experiment means: ", s_means)
    print("Experiment stderrs: ", s_stds)
    print("Permuted means: ", g_means)
    print("Permuted stderrs: ", g_stds)

    # Bootstrapped confidence interval for overcombine deviations
    bootstrap_oc_means = []
    for _ in range(5000):
        bootstrap_sample = np.random.choice(
            summary_matrix[2, :], size=int(len(summary_matrix[2, :]))
            )
        bootstrap_oc_means.append(np.mean(bootstrap_sample))
    ocCI = np.quantile(bootstrap_oc_means, [0.025, 0.975])
    print("Confidence interval overcombine: ", ocCI)

    # Get statistical significance for Fig10 bars
    # (except overcombine), compared to permuted data
    if permutations_for_quantiles is not None:
        for k_idx, k in enumerate(ks):
            if k_idx == 2:  # Have CI above
                continue
            q = scipy.stats.percentileofscore(
                permutations_for_quantiles[k], s_means[k_idx]
                )
            print(k, "quantile : ", q)

    # Bar plot, Fig 10G
    fig, ax = plt.subplots(1, figsize=(3, 3))
    bar_colors = {
        "absent": "#D7EDFB",
        "oversegment": "#FCDCD9",
        "overcombine": "#FEF0DD",
        "no_deviations": " #E5E5E5",
        "no_deviations_and_most_common": "#E5E5E5"
        }
    for k_idx, k in enumerate(ks):
        ax.bar(
            [k_idx],
            s_means[k_idx],
            color=bar_colors[k],
            edgecolor="black",
            label="__nolabel__"
            )
        ax.errorbar(
            [k_idx],
            s_means[k_idx],
            s_stds[k_idx],
            fmt=" ",
            ecolor="black",
            capsize=3,
            alpha=0.7
            )
        if k != "overcombine":
            ax.errorbar(
                [k_idx],
                g_means[k_idx],
                g_stds[k_idx],
                c="black",
                fmt=".",
                capsize=3,
                label="Permutations" if k_idx == 0 else "__nolabel__"
                )

    ax.set_xticks(
        range(len(ks)),
        [
            "Absent", "Oversegment", "Overcombine",
            "No deviations", "No deviations+\nMost common"
        ],
        fontsize="small", rotation=30, rotation_mode="anchor", ha="right"
        )
    ax.set_ylabel("Proportion of\npre-mixture recordings")
    plt.legend(fontsize="small")
    plt.tight_layout()
    ax.set_title(f"{full_experiment_name}, n={n_workers}", fontsize="xx-small")
    print(experiment_folder + "summary_stats.png")
    plt.savefig(experiment_folder + "summary_stats.png")
    plt.savefig(experiment_folder + "summary_stats.svg")
    plt.close()


def count_deviations(scene_idxs, response_matrices, experiment_results, mean_only=False, common_answers=None, all_common_answers=None):
    """ Count scene analysis deviations from pre-mixture sounds,
        given a dict of response matrices.
    """

    summary_statistics = {
        "absent": [],
        "oversegment": [],
        "overcombine": [],
        "no_deviations": []
        }
    if common_answers is not None or all_common_answers is not None:
        summary_statistics["no_deviations_and_most_common"] = []

    deviation_rate_per_recording = {}
    for scene_idx in scene_idxs:

        if experiment_results is not None:
            scene_entry = experiment_results.loc[experiment_results.exemplar == scene_idx]
            scene_entry = scene_entry.iloc[0]
        # scene_responses.shape: n_workers x row x col
        scene_responses = response_matrices[scene_idx]

        # Count up the different kinds of deviations
        # There is at least one check in each row because
        # they can't move on in the trial otherwise.
        # Multiple checks in a row means an undersegmentation deviation
        # overcombine_idxs.shape: nw x row
        overcombine_idxs = np.sum(scene_responses, axis=2) > 1

        # Count 1 deviation for each source that is involved
        # in the overcombination.
        overcombine_deviations = scene_responses.copy()
        overcombine_deviations[~overcombine_idxs, :] = 0
        # overcombine_deviations.shape: #w x col
        overcombine_deviations = 1.0*np.any(overcombine_deviations, axis=1)

        # Zero checks in a column means an absent deviation
        # absent_deviations.shape: nw x col
        absent_deviations = np.sum(scene_responses, axis=1) == 0

        # Multiple checks in a column means an oversegmentation deviation
        # If there are two checks, 1 oseg deviation.
        # If three, count as 2 oseg deviations. etc.
        # oversegment_deviations.shape = #nw x col
        oversegment_deviations = 1.0*(np.sum(scene_responses, axis=1) > 1)

        # Count as no deviations, if there aren't any of the other deviations
        no_deviations = (
            absent_deviations +
            oversegment_deviations +
            overcombine_deviations
            ) == 0

        # Compile by worker for summary plot
        summary_statistics["absent"].append(absent_deviations)
        summary_statistics["oversegment"].append(oversegment_deviations)
        summary_statistics["overcombine"].append(overcombine_deviations)
        summary_statistics["no_deviations"].append(no_deviations)
        if common_answers is not None:
            most_common_answer = np.concatenate([
                common_answers[scene_idx][col_idx][0][0]
                for col_idx in range(scene_responses.shape[2])],
                axis=1)
            gave_same_response_for_column = (
                scene_responses == most_common_answer[None, :, :]
                ).all(1)
            no_deviations_and_most_common = gave_same_response_for_column * no_deviations
            summary_statistics["no_deviations_and_most_common"].append(no_deviations_and_most_common)
        elif all_common_answers is not None:
            most_common_answer = np.stack(np.concatenate([
                all_common_answers[sim_idx][scene_idx][col_idx][0][0]
                for col_idx in range(scene_responses.shape[2])
                ], axis=1) for sim_idx in range(len(all_common_answers)))
            gave_same_response_for_column = (scene_responses == most_common_answer).all(1)
            no_deviations_and_most_common = gave_same_response_for_column * no_deviations
            summary_statistics["no_deviations_and_most_common"].append(no_deviations_and_most_common)

        # Compile by pre-mixture recording
        nw = absent_deviations.shape[0]
        for col_idx in range(scene_responses.shape[2]):
            cidx = scene_entry["col_source_idxs"][col_idx]
            deviation_rate_per_recording[(scene_idx, cidx)] = {}
            deviation_rate_per_recording[(scene_idx, cidx)]["absent"] = (
                np.mean(absent_deviations[:, col_idx]),
                np.std(absent_deviations[:, col_idx])/np.sqrt(nw)
                )
            deviation_rate_per_recording[(scene_idx, cidx)]["oversegment"] = (
                np.mean(oversegment_deviations[:, col_idx]),
                np.std(oversegment_deviations[:, col_idx])/np.sqrt(nw)
                )
            deviation_rate_per_recording[(scene_idx, cidx)]["overcombine"] = (
                np.mean(overcombine_deviations[:, col_idx]),
                np.std(overcombine_deviations[:, col_idx])/np.sqrt(nw)
                )
            deviation_rate_per_recording[(scene_idx, cidx)]["all"] = np.mean(absent_deviations[:, col_idx]) + np.mean(oversegment_deviations[:, col_idx]) + np.mean(overcombine_deviations[:, col_idx])
            deviation_rate_per_recording[(scene_idx, cidx)]["no_deviations"] = (
                np.mean(no_deviations[:, col_idx]),
                np.std(no_deviations[:, col_idx])
                )
            if common_answers is not None or all_common_answers is not None:
                deviation_rate_per_recording[(scene_idx, cidx)]["no_deviations_and_most_common"] = (
                    np.mean(no_deviations_and_most_common[:, col_idx]),
                    np.std(no_deviations_and_most_common[:, col_idx])
                    )

    for k, v in summary_statistics.items():
        # Concatenate over scenes
        v = np.concatenate(v, axis=1)
        # Mean over scenes, shape: (n_workers,)
        v = np.mean(v, axis=1)
        if mean_only:
            summary_statistics[k] = np.mean(v)
        else:
            # Standard error across scenes
            summary_statistics[k] = (np.mean(v), np.std(v)/np.sqrt(len(v)))

    return deviation_rate_per_recording, summary_statistics


def plot_deviations(deviation_rate_per_recording, experiment_folder, mixture_info=None):
    """ Plot scene analysis deviations
        bar graph for each type of deviations over scenes
        Figure F.2
    """

    deviation_type_order = ["absent", "oversegment", "overcombine"]
    ks = sorted(deviation_rate_per_recording.keys())
    x = range(len(ks))

    # Left side of Figure F.2
    bar_order = sorted(
        x, key=lambda _x: -deviation_rate_per_recording[ks[_x]]["all"]
        )
    fig = plt.figure(figsize=(8, 16))
    ax = fig.add_axes((0.6, 0.05, 0.35, 0.90))
    deviation_type_order = ["absent", "oversegment", "overcombine"]
    left = [0 for _ in x]
    # Plot the deviation types as stacked bars
    for _, deviation_type in enumerate(deviation_type_order):
        y = [
            deviation_rate_per_recording[ks[bidx]][deviation_type][0]
            for bidx in bar_order
            ]
        ax.barh(x, y, left=left, label=deviation_type)
        left = [_b + _y for _b, _y in zip(left, y)]
    xt = ["-".join([str(ks[bidx][0]), str(ks[bidx][1])]) for bidx in bar_order]
    # Label the bars with the category information
    categories = []
    for eidx, bidx in enumerate(bar_order):
        category = mixture_info[ks[bidx][0]][ks[bidx][1]]
        categories.append(category)
        xt[eidx] = category + ":" + xt[eidx]

    ax.set_yticks(x, xt, fontsize="xx-small")
    ax.set_ylim([-1, len(xt)+2])
    ax.set_ylabel("Premixture recording", fontsize="large")
    ax.set_xlabel("Mean # of deviations across workers")
    ax.set_title("Deviations by premixture sounds")
    ax.legend(deviation_type_order)
    print(experiment_folder + "stacked_deviations_by_source_vert.png")
    plt.savefig(experiment_folder + "stacked_deviations_by_source_vert.png")
    plt.savefig(experiment_folder + "stacked_deviations_by_source_vert.svg")
    plt.close()

    # Right side of Figure F.2
    cat_idxs = np.argsort(categories)
    sorted_cat = [categories[cat_idx] for cat_idx in cat_idxs]
    new_bar_order = [bar_order[bidx] for bidx in cat_idxs]
    fig = plt.figure(figsize=(8, 16))
    ax = fig.add_axes((0.6, 0.05, 0.35, 0.90))
    deviation_type_order = ["absent", "oversegment", "overcombine"]
    left = [0 for _ in y]
    for _, deviation_type in enumerate(deviation_type_order):
        y = [
            deviation_rate_per_recording[ks[bidx]][deviation_type][0]
            for bidx in new_bar_order
            ]
        ax.barh(x, y, left=left, label=deviation_type)
        left = [_b + _y for _b, _y in zip(left, y)]
    xt = [
        sorted_cat[eidx] +
        ":" +
        "-".join([str(ks[bidx][0]), str(ks[bidx][1])])
        for eidx, bidx in enumerate(new_bar_order)
        ]
    ax.set_yticks(x, xt, fontsize="xx-small")
    ax.set_ylim([-1, len(xt)+2])
    ax.set_ylabel("Premixture recording", fontsize="large")
    ax.set_xlabel("Mean # of deviations across workers")
    ax.set_title("Deviations by premixture sounds")
    ax.legend(deviation_type_order, fontsize="x-small")
    print(os.path.join(
        experiment_folder, "stacked_deviations_by_source_vert_bycat.png"
        ))
    plt.savefig(os.path.join(
        experiment_folder,
        "stacked_deviations_by_source_vert_bycat.png"
        ))
    plt.savefig(os.path.join(
        experiment_folder,
        "stacked_deviations_by_source_vert_bycat.svg"
        ))
    plt.close()

    return


def participant_deviations(experiment_results, workers_to_keep, experiment_folder, mixture_info=None, trial_info=None, n_splits=10):
    """ Plot the scene analysis deviations from premixture
        sounds that participants made on the sounds as a summary
    """

    # Collect the actual results from all valid workers
    scene_idxs = experiment_results.exemplar.unique()
    response_matrices = {}
    common_responses = {}
    for scene_idx in scene_idxs:
        scene_results = experiment_results.loc[(experiment_results.exemplar == scene_idx)]
        response_matrices[scene_idx] = np.stack(scene_results["response_matrix"])
        n_rows = response_matrices[scene_idx].shape[1]
        n_cols = response_matrices[scene_idx].shape[2]
        common_responses[scene_idx] = {}
        for col_idx in range(n_cols):
            response_counter = Counter([
                response_matrices[scene_idx][widx, :, col_idx].tobytes()
                for widx in range(response_matrices[scene_idx].shape[0])
                ])
            response_counter = [
                (np.frombuffer(k).reshape(n_rows, 1), v)
                for k, v in response_counter.items()
                ]
            common_responses[scene_idx][col_idx] = sorted(
                response_counter, key=lambda tup: tup[1], reverse=True
            )

    # Once you have all the scenes, compute error rates -- Figure F.2
    deviation_rate_per_recording, _ = count_deviations(
        scene_idxs,
        response_matrices,
        experiment_results
        )
    plot_deviations(
        deviation_rate_per_recording, experiment_folder,
        mixture_info=mixture_info
        )

    # Now per worker -- this will end up in Fig 10.G
    summary_stats_per_worker = {}
    for i, worker_idx in enumerate(workers_to_keep):
        worker_response_matrix = {}
        for scene_idx in scene_idxs:
            scene_results = experiment_results.loc[
                (experiment_results.exemplar == scene_idx) &
                (experiment_results.worker_idx == worker_idx)
                ]
            worker_response_matrix[scene_idx] = scene_results["response_matrix"].item()[None, :, :]
        _, worker_summary_statistics = count_deviations(
            scene_idxs,
            worker_response_matrix,
            experiment_results,
            mean_only=True,
            common_answers=common_responses
            )
        summary_stats_per_worker[worker_idx] = worker_summary_statistics

    # Split half reliability analysis (Figure F.1, bars)
    print("Split half realibility, for actual data")
    split_half_analysis = []
    for simulation in tqdm(range(n_splits)):
        this_simulation_split = np.random.permutation(len(workers_to_keep))
        split1 = this_simulation_split[:(len(this_simulation_split)//2)]
        split2 = this_simulation_split[(len(this_simulation_split)//2):]
        # First split the response matrices
        split1_response_matrices = {}
        split1_common_response = {}
        split2_response_matrices = {}
        split2_common_response = {}
        for scene_idx in response_matrices.keys():
            split1_response_matrices[scene_idx] = response_matrices[scene_idx][split1, :, :]
            split2_response_matrices[scene_idx] = response_matrices[scene_idx][split2, :, :]
            n_cols = trial_info.loc[trial_info.exemplar == scene_idx]["n_recordings"].item()
            n_rows = trial_info.loc[trial_info.exemplar == scene_idx]["n_rowsounds"].item()
            split1_common_response[scene_idx] = {}
            split2_common_response[scene_idx] = {}
            for col_idx in range(n_cols):
                # Split1
                response_counter = Counter([
                    split1_response_matrices[scene_idx][widx, :, col_idx].tobytes()
                    for widx in range(split1_response_matrices[scene_idx].shape[0])
                    ])
                response_counter = [
                    (np.frombuffer(k).reshape(n_rows, 1), v)
                    for k, v in response_counter.items()
                    ]
                split1_common_response[scene_idx][col_idx] = sorted(
                    response_counter,
                    key=lambda tup: tup[1],
                    reverse=True
                    )
                # Split2
                response_counter = Counter([
                    split2_response_matrices[scene_idx][widx, :, col_idx].tobytes()
                    for widx in range(split2_response_matrices[scene_idx].shape[0])
                    ])
                response_counter = [
                    (np.frombuffer(k).reshape(n_rows, 1), v)
                    for k, v in response_counter.items()
                    ]
                split2_common_response[scene_idx][col_idx] = sorted(
                    response_counter,
                    key=lambda tup: tup[1],
                    reverse=True
                    )

        # Once you have all the scenes, compute error rates
        deviation_rate_per_recording_sh1, _ = count_deviations(
            scene_idxs,
            split1_response_matrices,
            experiment_results,
            mean_only=True,
            common_answers=split1_common_response
            )
        deviation_rate_per_recording_sh2, _ = count_deviations(
            scene_idxs,
            split2_response_matrices,
            experiment_results,
            mean_only=True,
            common_answers=split2_common_response
            )
        split_half_analysis.append((
            deviation_rate_per_recording_sh1, deviation_rate_per_recording_sh2
            ))

    devtypes = ["absent", "oversegment", "overcombine", "no_deviations", "no_deviations_and_most_common"]
    rs = {k: [] for k in devtypes}
    for simulation in range(n_splits):
        split1stats = split_half_analysis[simulation][0]
        split2stats = split_half_analysis[simulation][1]
        for deviation_type in devtypes:
            x = []
            y = []
            for premixture_id in split1stats.keys():
                x.append(split1stats[premixture_id][deviation_type][0])
                y.append(split2stats[premixture_id][deviation_type][0])
            r, _ = scipy.stats.pearsonr(x, y)
            rs[deviation_type].append(r)

    # Fig. F.1 bars
    print("Split half mean and std correlation:")
    for deviation_type in devtypes:
        print(deviation_type, np.mean(rs[deviation_type]), np.std(rs[deviation_type]))

    return summary_stats_per_worker


def random_simulations(experiment_results, worker_idxs, trial_info, n_simulations, n_splits=10):
    """ Compute baseline result from permuting the participant data """

    # Get the indexes of mixtures in the experiment
    scene_idxs = experiment_results["exemplar"].unique()

    # Raters need to put at least one check per row
    # Count how often they are doing more than that
    worker_k = []
    worker_N = []
    for worker_idx in worker_idxs:
        worker_results = experiment_results.loc[
            experiment_results.worker_idx == worker_idx
            ]
        ps = []
        k = 0
        N = 0
        for response_matrix in worker_results.response_matrix:
            row_sum_minus_1 = (response_matrix.sum(1) - 1).sum()
            leftover = (response_matrix.shape[0] *
                        response_matrix.shape[1] -
                        response_matrix.shape[0])
            if leftover > 0:
                ps.append(row_sum_minus_1/leftover)
            k += row_sum_minus_1
            N += leftover
        worker_k.append(k)
        worker_N.append(N)

    # Given those responding rates, let's simulate responses
    all_simulated_response_matrices = []
    all_simulated_common_responses = []
    worker_across_simulations = [
        {k: [] for k in scene_idxs}
        for _ in worker_idxs
        ]
    print("Starting simulation...")
    for simulation in tqdm(range(n_simulations)):
        sim_worker_k = list(worker_k)
        sim_worker_N = list(worker_N)
        response_matrices = {}
        common_responses = {}
        for scene_idx in scene_idxs:
            # Get col and row to simulate a matrix of the right size
            n_cols = trial_info.loc[trial_info.exemplar == scene_idx]["n_recordings"].item()
            n_rows = trial_info.loc[trial_info.exemplar == scene_idx]["n_rowsounds"].item()
            random_responses = []
            # Simulate each worker, with one check per row
            # and their beyond1 rate
            for i, worker_idx in enumerate(worker_idxs):
                # Sample the required single row checkmark
                row_checks = np.random.randint(n_cols, size=n_rows)
                # Sample more checkmarks depending on the worker_rate
                n_more_boxes = n_rows*n_cols - n_rows
                # permute
                n_more_checks = np.random.hypergeometric(
                    sim_worker_k[i],
                    sim_worker_N[i] - sim_worker_k[i],
                    n_more_boxes
                    )
                sim_worker_k[i] -= n_more_checks
                sim_worker_N[i] -= n_more_boxes
                # Bank of extra check marks
                extra_checks = np.zeros(n_rows*n_cols - n_rows)
                extra_checks[:n_more_checks] = 1
                extra_checks = np.random.permutation(extra_checks)
                # Place the checkmarks
                random_response = np.zeros((n_rows, n_cols))
                for row_idx in range(n_rows):
                    random_response[row_idx, row_checks[row_idx]] = 1
                extra_checks_idx = 0
                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        if col_idx == row_checks[row_idx]:
                            # Don't put extra checks where
                            # there is already a check
                            continue
                        else:
                            random_response[row_idx, col_idx] = extra_checks[extra_checks_idx]
                            extra_checks_idx += 1
                random_responses.append(random_response)
                worker_across_simulations[i][scene_idx].append(random_response)
            # Combine the responses for a single scene across workers
            responses_across_workers = np.stack(random_responses)
            response_matrices[scene_idx] = responses_across_workers

            # Get counts over the response placement to
            # determine the most common deviation
            common_responses[scene_idx] = {}
            for col_idx in range(n_cols):
                response_counter = Counter([
                    rr[:, col_idx].tobytes() for rr in random_responses
                    ])
                response_counter = [(
                    np.frombuffer(k).reshape(n_rows, 1), v
                    ) for k, v in response_counter.items()
                    ]
                common_responses[scene_idx][col_idx] = sorted(
                    response_counter, key=lambda tup: tup[1], reverse=True
                    )
        all_simulated_response_matrices.append(response_matrices)
        all_simulated_common_responses.append(common_responses)

    # Stack across simulations
    for i, worker_idx in enumerate(worker_idxs):
        for scene_idx in scene_idxs:
            worker_across_simulations[i][scene_idx] = np.stack(
                worker_across_simulations[i][scene_idx]
                )

    # In a list by worker, number of deviations averaged across trials and simulations
    summary_stats_by_worker = {
        worker_idx: count_deviations(
            scene_idxs,
            worker_across_simulations[i],
            experiment_results,
            mean_only=True,
            all_common_answers=all_simulated_common_responses
        )[1] for i, worker_idx in enumerate(worker_idxs)
        }

    simulated_rates = []
    devtypes = [
        "absent",
        "oversegment",
        "overcombine",
        "no_deviations",
        "no_deviations_and_most_common"
        ]
    summary_statistics = {devtype: [] for devtype in devtypes}
    for simulation in range(n_simulations):
        # Once you have all the scenes, compute error rates
        deviation_rate_per_recording, this_sim_summary_stats = count_deviations(
            scene_idxs,
            all_simulated_response_matrices[simulation],
            experiment_results,
            mean_only=True,
            common_answers=all_simulated_common_responses[simulation]
            )
        for stat in summary_statistics.keys():
            summary_statistics[stat].append(this_sim_summary_stats[stat])
        simulated_rates.append(deviation_rate_per_recording)

    # Split half reliability analysis
    print("Split half reliability for simulation")
    split_half_analysis = []
    n_subsampled_simulations = 100
    subsample_simulations_for_split_half = np.random.permutation(n_simulations)[:n_subsampled_simulations]
    for simulation_idx in tqdm(range(n_subsampled_simulations)):
        simulation = subsample_simulations_for_split_half[simulation_idx]
        split_analysis = []
        for split_idx in range(n_splits):
            this_simulation_response_matrices = all_simulated_response_matrices[simulation]
            this_simulation_split = np.random.permutation(len(worker_idxs))
            split1 = this_simulation_split[:(len(this_simulation_split)//2)]
            split2 = this_simulation_split[(len(this_simulation_split)//2):]
            # First split the response matrices
            split1_response_matrices = {}
            split1_common_response = {}
            split2_response_matrices = {}
            split2_common_response = {}
            for scene_idx in this_simulation_response_matrices.keys():
                split1_response_matrices[scene_idx] = all_simulated_response_matrices[simulation][scene_idx][split1, :, :]
                split2_response_matrices[scene_idx] = all_simulated_response_matrices[simulation][scene_idx][split2, :, :]
                n_cols = trial_info.loc[trial_info.exemplar == scene_idx]["n_recordings"].item()
                n_rows = trial_info.loc[trial_info.exemplar == scene_idx]["n_rowsounds"].item()
                split1_common_response[scene_idx] = {}
                split2_common_response[scene_idx] = {}
                for col_idx in range(n_cols):
                    # Split1
                    response_counter = Counter([
                        split1_response_matrices[scene_idx][widx, :, col_idx].tobytes()
                        for widx in range(split1_response_matrices[scene_idx].shape[0])
                        ])
                    response_counter = [
                        (np.frombuffer(k).reshape(n_rows, 1), v)
                        for k, v in response_counter.items()
                        ]
                    split1_common_response[scene_idx][col_idx] = sorted(
                        response_counter, key=lambda tup: tup[1], reverse=True
                        )
                    # Split2
                    response_counter = Counter([
                        split2_response_matrices[scene_idx][widx, :, col_idx].tobytes()
                        for widx in range(split2_response_matrices[scene_idx].shape[0])
                        ])
                    response_counter = [
                        (np.frombuffer(k).reshape(n_rows, 1), v)
                        for k, v in response_counter.items()
                        ]
                    split2_common_response[scene_idx][col_idx] = sorted(
                        response_counter, key=lambda tup: tup[1], reverse=True
                        )

            # Once you have all the scenes, compute error rates
            deviation_rate_per_recording_sh1, _ = count_deviations(
                scene_idxs,
                split1_response_matrices,
                experiment_results,
                mean_only=True,
                common_answers=split1_common_response
                )
            deviation_rate_per_recording_sh2, _ = count_deviations(
                scene_idxs,
                split2_response_matrices,
                experiment_results,
                mean_only=True,
                common_answers=split2_common_response
                )
            split_analysis.append((
                deviation_rate_per_recording_sh1,
                deviation_rate_per_recording_sh2
                ))

        split_half_analysis.append(split_analysis)

    rs = {k: [] for k in devtypes}
    for simulation_idx in tqdm(range(n_subsampled_simulations)):
        simulation = subsample_simulations_for_split_half[simulation_idx]
        splitrs = {k: [] for k in devtypes}
        for split_idx in range(n_splits):
            split1stats = split_half_analysis[simulation][split_idx][0]
            split2stats = split_half_analysis[simulation][split_idx][1]
            for deviation_type in devtypes:
                x = []
                y = []
                for premixture_id in split1stats.keys():
                    x.append(split1stats[premixture_id][deviation_type][0])
                    y.append(split2stats[premixture_id][deviation_type][0])
                r, _ = scipy.stats.pearsonr(x, y)
                splitrs[deviation_type].append(r)
            for deviation_type in devtypes:
                rs[deviation_type].append(np.mean(splitrs[deviation_type]))

    # Fig F.1 permutations
    print("Simulated split half analysis, mean and std correlation:")
    for deviation_type in devtypes:
        print(
            deviation_type,
            np.mean(rs[deviation_type]),
            np.std(rs[deviation_type])
            )

    return summary_statistics, summary_stats_by_worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-name", help="name of results file")
    parser.add_argument("--expt-name",
                        help="name of experiment of model inferences")
    parser.add_argument("--sound-group",
                        help="name of sounds for which we did model inference")
    parser.add_argument("--expt-idx", type=int,
                        help="numbering of experiment with this \
                              experiment name and sound group")
    args = parser.parse_args()
    main(args.csv_name, args.expt_name, args.sound_group, args.expt_idx)
