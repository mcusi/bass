import os
import yaml
import argparse
import numpy as np
import pandas as pd

import statsmodels.stats.api as sms
import matplotlib.pyplot as plt


def main(csv_name, expt_name, sound_group, expt_idx, catch_trial_threshold):
    """ Analyzes csv returned from Experiment 1 """

    # Load in results CSV and experiment config
    raw_results = pd.read_csv(os.path.join(
        os.environ["home_dir"], "audio", "results", csv_name + ".csv"
        ))
    full_experiment_name = "-".join([expt_name, sound_group, f"{expt_idx:03d}"])
    experiment_folder = os.path.join(
        os.environ["sound_dir"], sound_group,
        "experiments", full_experiment_name, ""
        )
    with open(experiment_folder + "config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Loop through each participant in the experiment to collect answers
    trial_table = []
    table_columns = [
        "worker_idx", "condition", "correct", "original_mixture",
        "sound_number", "split", "original_mixture_idx", "source_idx"
        ]
    n_workers = len(raw_results["WorkerId"])
    workers_to_skip = []
    headphone_check_failed = []
    workers_to_keep = []
    for worker_idx in range(n_workers):

        # Check if participant failed the headphone check
        if raw_results["Answer.HC_correct"][worker_idx] < 5:
            workers_to_skip.append(worker_idx)
            headphone_check_failed.append(worker_idx)
            continue
        # Was this participant in split00 or split01?
        split_str = raw_results["Answer.split"][worker_idx]
        if isinstance(split_str, str):
            which_split = int(split_str[0])
        else:
            workers_to_skip.append(worker_idx)
            continue

        # Get trial info for this worker
        trial_info_for_split = pd.read_csv(os.path.join(
            experiment_folder, f"trial_split{which_split:02d}_info.csv"
            ))
        # Basic info to reconstruct what happened
        # Can compare to trial_info_for_split
        trial_conditions = raw_results["Answer.condition"][worker_idx].split("|")
        correct_responses = raw_results["Answer.trial_order"][worker_idx].split("|")
        soundnums_played = raw_results["Answer.soundnum_for_trial"][worker_idx].split("|")

        # Set up catch trial checks
        worker_table = []
        if config["n_catch_trials"] > 0:
            # Catch trials will always be the first (N=n_catch_trials)
            # sound numbers, although the order in the trials will be
            # randomized
            catch_trial_results = []

            def add_to_worker_table(sound_number):
                return sound_number >= config["n_catch_trials"]

        else:
            def add_to_worker_table(_):
                return True

        for trial_idx in range(len(trial_conditions)):
            # Figure out whether participant gave a
            # correct answer on this trial
            condition = "recorded" if int(trial_conditions[trial_idx]) == 0 else "model"
            actual_response = int(
                raw_results[f"Answer.resp_for_trial_{trial_idx}"][worker_idx]
                )
            correct_response = int(
                correct_responses[trial_idx] == "true"
                )
            worker_responded_correctly = actual_response == correct_response
            # Determine which mixture sound was played
            soundnum_played = int(soundnums_played[trial_idx])
            original_mixture = trial_info_for_split.loc[
                trial_info_for_split.sound_number == soundnum_played
                ]["scene_dest"].item()
            original_mixture_idx = int(
                os.path.split(original_mixture)[-1].split("_")[1]
                )
            inferred_source_idx = trial_info_for_split.loc[
                trial_info_for_split.sound_number == soundnum_played
                ]["source_idx"].item()
            if add_to_worker_table(soundnum_played):
                worker_table.append([
                    worker_idx, condition, worker_responded_correctly,
                    original_mixture, soundnum_played, which_split,
                    original_mixture_idx, inferred_source_idx
                    ])
            else:
                catch_trial_results.append(worker_responded_correctly)

        # Throw out participant if they do not meet the catch_trial_threshold
        if config["n_catch_trials"] > 0:
            if np.mean(catch_trial_results) < catch_trial_threshold:
                workers_to_skip.append(worker_idx)
            else:
                workers_to_keep.append((worker_idx, which_split))
                trial_table.extend(worker_table)
        else:
            workers_to_keep.append((worker_idx, which_split))
            trial_table.extend(worker_table)

    # Save sorted results
    print("N workers kept: ", len(workers_to_keep))
    print("N failed headphone check: ", len(headphone_check_failed))
    print(f"{len(workers_to_skip)} workers to exclude: ", workers_to_skip)
    experiment_results = pd.DataFrame(trial_table, columns=table_columns)
    experiment_results.to_csv(experiment_folder + "results.csv")
    # Analyses and plots
    get_demographics(raw_results, workers_to_keep)
    plot_by_condition(
        experiment_results,
        n_workers,
        full_experiment_name,
        experiment_folder,
        workers_to_skip
        )
    for condition in ["model", "recorded"]:
        plot_quartiles(
            experiment_results,
            n_workers - len(workers_to_skip),
            full_experiment_name,
            experiment_folder,
            condition
            )


def get_demographics(raw_results, workers_to_keep):
    """ Collect demographics of age and gender for
        workers included in the experiment
    """
    genders = {"female": 0, "male": 0, "non-binary": 0, "other": 0}
    ages = []
    for worker_idx, _ in workers_to_keep:
        gender = raw_results.iloc[worker_idx]["Answer.Q1Gender"]
        age = raw_results.iloc[worker_idx]["Answer.Q2age"]
        ages.append(int(age))
        if gender[0].lower() == "f":
            genders["female"] += 1
        elif gender[0].lower() == "m":
            genders["male"] += 1
        elif "non" in gender:
            genders["non-binary"] += 1
        else:
            print("Other gender:", gender)
            genders["other"] += 1
    print("Gender: ", genders)
    print("Age mean: ", np.mean(ages))
    print("Age std: ", np.std(ages))


def plot_by_condition(experiment_results, n_workers, full_experiment_name, experiment_folder, workers_to_skip):
    """ Figure 10B: Gives confidence intervals and bar plot for
        accuracy in Expt 1 by the two conditions, model and recorded
    """

    accuracy_recorded = []
    accuracy_model = []
    for worker_idx in range(n_workers):
        if worker_idx in workers_to_skip:
            continue
        # Results for recorded audio
        worker_recorded_rows = experiment_results.loc[
            (experiment_results["worker_idx"] == worker_idx) &
            (experiment_results["condition"] == "recorded")
            ]
        accuracy_recorded.append(np.mean(worker_recorded_rows["correct"]))
        # Results for model audio
        worker_model_rows = experiment_results.loc[
            (experiment_results["worker_idx"] == worker_idx) &
            (experiment_results["condition"] == "model")
            ]
        accuracy_model.append(np.mean(worker_model_rows["correct"]))

    # Accuracy and standard error for recorded audio
    mean_recorded = np.mean(accuracy_recorded)
    stderr_recorded = np.std(accuracy_recorded)/np.sqrt(len(accuracy_recorded))
    # Accuracy and standard error for model audio
    mean_model = np.mean(accuracy_model)
    stderr_model = np.std(accuracy_model)/np.sqrt(len(accuracy_model))
    # Confidence intervals for both conditions
    confidence_interval_recorded = sms.DescrStatsW(
        accuracy_recorded
        ).tconfint_mean()
    print("Confidence interval recorded: ", confidence_interval_recorded)
    confidence_interval_model = sms.DescrStatsW(accuracy_model).tconfint_mean()
    print("Confidence interval model: ", confidence_interval_model)

    # Figure 10B
    x = [1, 2]
    y = [mean_recorded, mean_model]
    plt.figure(figsize=(2, 2))
    plt.bar(x, y, color="silver", edgecolor="black")
    plt.errorbar(x, y, yerr=[stderr_recorded, stderr_model], fmt=" ", c="black", capsize=3)
    plt.plot([0.5, 2.5], [0.50, 0.50], "k--", alpha=0.5)
    for aidx, ar in enumerate(accuracy_recorded):
        plt.plot(x, [ar, accuracy_model[aidx]], alpha=0.05, c="black")
    plt.xticks(x, ["Recorded", "Model"])
    plt.title(
        full_experiment_name +
        f"\nn={n_workers - len(workers_to_skip)}, \
        excl={len(workers_to_skip)}",
        fontsize="x-small"
        )
    plt.xlabel("Sound condition")
    plt.ylabel("Proportion correct")
    plt.ylim([0, 1])
    plt.xlim([0.5, 2.5])
    plt.tight_layout()
    print(os.path.join(experiment_folder, "proportion_correct.png"))
    plt.savefig(os.path.join(experiment_folder, "proportion_correct.png"))
    plt.savefig(os.path.join(experiment_folder, "proportion_correct.svg"))
    plt.close()


def plot_quartiles(experiment_results, n_workers, full_experiment_name, experiment_folder, condition, recognizability_threshold=0.6):
    """ Figure 10C: create a histogram the recognizability of the model sounds in Expt 1 """

    split_sns = []
    results = {}
    n_participants_dict = {}
    for split_idx in range(2):
        sound_numbers_in_split = experiment_results.loc[
            (experiment_results["split"] == split_idx) &
            (experiment_results["condition"] == condition)
            ]["sound_number"].unique()
        split_sns.append(sound_numbers_in_split)
        for sn in sound_numbers_in_split:
            this_trial_results = experiment_results.loc[
                (experiment_results["split"] == split_idx) &
                (experiment_results["condition"] == condition) &
                (experiment_results["sound_number"] == sn)
                ]
            source_idx = this_trial_results["source_idx"].head(1).item()
            scene_number = this_trial_results["original_mixture_idx"].head(1).item()
            results[(split_idx, sn, str(scene_number), str(source_idx))] = np.mean(this_trial_results["correct"])
            n_participants_dict[(split_idx, sn, str(scene_number), str(source_idx))] = len(this_trial_results["correct"])

    # Figure 10C
    sorted_trials = sorted(results.items(), key=lambda i: i[1])
    plt.figure(figsize=(2, 2))
    plt.hist([st[1] for st in sorted_trials], color="silver")
    plt.xlim([0, 1])
    plt.xlabel("Proportion correct")
    plt.ylabel("Number of sources")
    plt.title(
        full_experiment_name +
        f"\ncondition={condition}, n={n_workers}",
        fontsize="xx-small"
        )
    print(os.path.join(experiment_folder, f"hist_by_{condition}_trial.png"))
    plt.tight_layout()
    plt.savefig(os.path.join(
        experiment_folder, f"hist_by_{condition}_trial.png"
        ))
    plt.savefig(os.path.join(
        experiment_folder, f"hist_by_{condition}_trial.svg"
        ))
    plt.close()

    # Record which sounds are unrecognizable
    # because these will be excluded in Experiment 2
    with open(os.path.join(experiment_folder, f"unrecognizable_{condition}.txt"), "w") as f:
        print(f"Condition: {condition}, \
                n_trials:{len(sorted_trials)}, \
                n_unrecognizable:{sum([st[1]<recognizability_threshold for st in sorted_trials])}, \
                n_recognizable:{len(sorted_trials) - sum([st[1]<recognizability_threshold for st in sorted_trials])}")
        for st in sorted_trials:
            if st[1] < recognizability_threshold:
                f.write(f"{st[0][0]},{st[0][1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-name", help="name of results file")
    parser.add_argument("--expt-name",
                        help="name of experiment of model inferences"
                        )
    parser.add_argument("--sound-group",
                        help="name of sounds for which we did model inference"
                        )
    parser.add_argument("--expt-idx", type=int,
                        help="numbering of experiment with this experiment name and sound group"
                        )
    parser.add_argument("--catch-trial_threshold", default=0.7, type=float,
                        help="participants must achieve an accuracy of at least this much to be included"
                        )
    args = parser.parse_args()
    main(
        args.csv_name,
        args.expt_name,
        args.sound_group,
        args.expt_idx,
        args.catch_trial_threshold
        )
