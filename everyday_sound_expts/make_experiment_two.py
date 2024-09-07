import os
import argparse
import json
import yaml
import pandas as pd
from glob import glob
import numpy as np
import soundfile as sf
from shutil import copyfile

import everyday_sound_expts.trials as trials_util


terminal_colors = {
    "RED": "\033[91m",
    "ENDC": "\033[0m"
    }


def main(expt_name, sound_group, expt_idx, trial_config):

    # Set up locations
    sound_folder = os.path.join(os.environ["sound_dir"], sound_group, "")
    with open(sound_folder + "config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(trial_config)
    print(config)
    mixture_info = pd.read_csv(os.path.join(sound_folder, "mixture_info.csv"))

    # Experiment 1 has some info we need,
    # like which sound recordings to leave out
    experiment1_folder = "-".join([expt_name, sound_group, f"{expt_idx:03d}"])
    experiment2_folder = "-".join([expt_name, sound_group, f"{expt_idx:03d}-expt2"])
    full_experiment1_path = os.path.join(
        sound_folder, "experiments", experiment1_folder, ""
        )
    full_experiment2_path = os.path.join(
        sound_folder, "experiments", experiment2_folder, ""
        )
    os.makedirs(full_experiment2_path)
    with open(full_experiment2_path + "config.yaml", 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)
    trials_path = os.path.join(
        full_experiment2_path, experiment2_folder + "_trials", ""
        )
    os.makedirs(trials_path, exist_ok=True)

    # Exclude the unrecognizable sources from expt 1
    scenes = sorted(glob(os.path.join(
        os.environ["sound_dir"], sound_group, "*scene.wav"
        )))
    unrecognizable_sources = {}
    for condition in ["recorded", "model"]:
        unrecognizable_sources[condition] = []
        with open(full_experiment1_path + f"unrecognizable_{condition}.txt", "r") as f:
            exlines = f.readlines()
            split_trial_pairs = []
            for exline in exlines:
                split_trial_pairs.append([
                    int(x) for x in exline[:-1].split(",")
                    ])
            for (split, trial) in split_trial_pairs:
                trial_info = pd.read_csv(os.path.join(
                    full_experiment1_path, f'trial_split{split:02d}_info.csv'
                    ))
                unrecognizable_trial = trial_info.iloc[trial]
                if condition == "recorded":
                    unrecognizable_sources[condition].append(
                        unrecognizable_trial["source_dest"]
                        )
                elif condition == "model":
                    unrecognizable_sources[condition].append((
                        unrecognizable_trial["source_dest"],
                        unrecognizable_trial["source_idx"]
                        ))
        print(terminal_colors["RED"] +
              f"{condition}: Skipped {len(unrecognizable_sources[condition])} \
              for unrecognizability." +
              terminal_colors["ENDC"]
              )

    # Get everyday sound trials
    # 1. Get catch trials
    full_catch_trial_path = os.path.join(full_experiment2_path, "catch_trials", "")
    rel_catch_trial_path = os.path.join(
        sound_group, "experiments", experiment2_folder, "catch_trials"
        )
    os.makedirs(full_catch_trial_path)
    catch_trial_info, catch_trial_config = trials_util.catch_expt2(
        config, mixture_info, experiment_folder=rel_catch_trial_path
        )
    catch_trial_scenes = sorted(glob(os.path.join(
        full_experiment2_path, "catch_trials", "*scene.wav"
        )))
    os.makedirs(os.path.join(full_catch_trial_path, "trials", ""))
    catch_trial_info, catch_trial_idx = trials_util.format_expt2(
        expt_name, sound_group, catch_trial_scenes,
        catch_trial_info, unrecognizable_sources,
        catch_trial_config,
        os.path.join(full_catch_trial_path, "trials", ""),
        mode="catch"
        )
    catch_trial_info.to_pickle(os.path.join(
        full_experiment2_path, 'catch_trial_info.pkl'
        ))
    # 2. Add experimental trials with model inferences
    expt_trial_info, _ = trials_util.format_expt2(
        expt_name, sound_group, scenes, mixture_info,
        unrecognizable_sources, config, trials_path,
        mode="full", sound_number=catch_trial_idx
        )
    expt_trial_info.to_pickle(os.path.join(
        full_experiment2_path, 'expt_trial_info.pkl'
        ))
    all_trial_info = pd.concat([catch_trial_info, expt_trial_info])
    all_trial_info.reset_index(drop=True, inplace=True)
    all_trial_info.to_pickle(os.path.join(
        full_experiment2_path, "all_trial_info.pkl"
        ))

    # These are required to be copied into the HTML
    # to determine the size of the tables for the trial
    n_cols_per_trial = all_trial_info["n_recordings"].tolist()
    n_rows_per_trial = all_trial_info["n_rowsounds"].tolist()
    print(os.path.join(full_experiment2_path, "n_cols_per_trial.json"))
    with open(os.path.join(full_experiment2_path, "n_cols_per_trial.json"), "w") as outfile:
        json.dump(n_cols_per_trial, outfile)
    with open(os.path.join(full_experiment2_path, "n_rows_per_trial.json"), "w") as outfile:
        json.dump(n_rows_per_trial, outfile)

    print("Adding catch trials to trial folder!")
    wavs_in_catch_trials = glob(os.path.join(full_catch_trial_path, "trials", "*.wav"))
    for w in wavs_in_catch_trials:
        copyfile(w, os.path.join(trials_path + os.path.basename(w)))

    # Amplitude normalize sounds
    # 1) Find the max amplitude sound
    print("Amplitude normalizing the sounds...")
    ms = []
    wavfns = glob(os.path.join(trials_path, "*.wav"))
    for wfn in wavfns:
        x, sr = sf.read(wfn)
        ms.append(np.max(np.abs(x)))
    mnorm = np.max(ms) * 1.001
    # 2) Normalize such that the max amplitude sound is set to ~1
    # (with some space not to clip)
    for wfn in wavfns:
        x, sr = sf.read(wfn)
        sf.write(wfn, x/mnorm, sr)

    print("Complete!")
    print("Check out experiment at: ", full_experiment2_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt-name", help="name of results file")
    parser.add_argument("--sound-group", help="name of results file")
    parser.add_argument("--expt-idx", type=int, help="name of results file")
    parser.add_argument("--n-catch-trials", type=int, default=10,
                        help="extra trials to assess whether participants are guessing"
                        )
    args = parser.parse_args()
    trial_config = {
        "n_catch_trials": args.n_catch_trials,
    }
    main(args.expt_name, args.sound_group, args.expt_idx, trial_config)
