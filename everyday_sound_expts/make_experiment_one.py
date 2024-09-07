import os
import yaml
import json
import random
import argparse
from glob import glob
from shutil import copyfile

import pandas as pd
import soundfile as sf
import numpy as np

import everyday_sound_expts.trials as trials_util
from util.sample import manual_seed


def main(expt_name, sound_group, foil_config):
    """ Creates a folder of sounds and experimental
        trials for experiment 1, given a set of
        inferred sources
    """

    sound_folder = os.path.join(os.environ["sound_dir"], sound_group, "")
    with open(sound_folder + "config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    manual_seed(config["seed"])
    mixture_info = pd.read_csv(sound_folder + "mixture_info.csv")
    config.update(foil_config)

    # Make experiment folder corresponding to this run of inference,
    # set of sounds and experiment settings.
    existing_expts_with_this_sound_group = glob(
        os.path.join(
            sound_folder,
            "experiments",
            "-".join([expt_name, sound_group, "*"]),
            ""
            )
        )
    this_expt_number = len(existing_expts_with_this_sound_group) + 1
    experiment_folder = "-".join([
        expt_name,
        sound_group,
        f"{this_expt_number:03d}"
        ])
    full_experiment_path = os.path.join(
        sound_folder, "experiments", experiment_folder, ""
        )
    os.makedirs(full_experiment_path)
    with open(full_experiment_path + "config.yaml", 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)
    trials_path = os.path.join(
        full_experiment_path, experiment_folder + "_trials", ""
        )
    os.makedirs(trials_path)

    # Split the audio files between everyday audio and model inferences
    scenes = sorted(glob(os.path.join(
        os.environ["sound_dir"], sound_group, "*scene.wav"
        )))
    n_scenes = len(scenes)
    scene_conds = [0 for _ in range(int(n_scenes/2))]  # recorded audio
    scene_conds.extend([1 for _ in range(n_scenes - len(scene_conds))])  # source audio
    random.shuffle(scene_conds)
    with open(full_experiment_path + "scene_conditions.json", "w") as outfile:
        json.dump(scene_conds, outfile)

    # Get everyday sound trials
    # 1. Get catch trials
    if config["n_catch_trials"] > 0:
        full_catch_trial_path = os.path.join(
            full_experiment_path, "catch_trials", ""
            )
        rel_catch_trial_path = os.path.join(
            sound_group, "experiments", experiment_folder, "catch_trials"
            )
        os.makedirs(full_catch_trial_path)
        catch_trial_info, catch_trial_config = trials_util.catch_expt1(
            config, mixture_info, experiment_folder=rel_catch_trial_path
            )

    # 2. Get foils for all everyday audio, whether catch trial or not
    recorded_foils_path = os.path.join(full_experiment_path, "recorded_foils", "")
    os.makedirs(recorded_foils_path, exist_ok=True)
    all_recorded_scenes_info = mixture_info if config["n_catch_trials"] == 0 else pd.concat([mixture_info, catch_trial_info])
    recorded_pairs_info = trials_util.recorded_foils_expt1(
        all_recorded_scenes_info, experiment_folder, sound_group, config
        )
    recorded_pairs_info.to_csv(
        os.path.join(full_experiment_path, 'all_recorded_pairs_info.csv')
    )

    # 3. Design catch trials which are meant to
    # be the same across both splits of scenes
    if config["n_catch_trials"] > 0:
        catch_trial_scenes = sorted(glob(os.path.join(
            full_experiment_path, "catch_trials", "*scene.wav"
            )))
        # Consider catch all as recorded sound trials, across both splits
        catch_trial_conds = [0 for _ in catch_trial_scenes]
        os.makedirs(os.path.join(full_catch_trial_path, "trials", ""))
        catch_trial_info, catch_trial_idx = trials_util.recorded_expt1(
            recorded_pairs_info,
            catch_trial_scenes,
            catch_trial_conds,
            catch_trial_config,
            os.path.join(full_catch_trial_path, "trials", ""),
            catch_trial=True
            )
        catch_trial_info.to_csv(os.path.join(
            full_experiment_path, 'catch_trial_info.csv'
            ))
    else:
        catch_trial_idx = 0

    print("Creating splits")
    for split in range(2):

        if split == 1:
            # Half of participants should get the reverse split
            scene_conds = [int(1.0*(sc == 0)) for sc in scene_conds]

        # Everyday sound trials
        # 2. for the scene_conds == 0, create everyday audio trials
        split_trials_path = os.path.join(trials_path, f"split{split:02d}", "")
        os.makedirs(split_trials_path, exist_ok=True)
        recorded_trial_info, recorded_trial_idx = trials_util.recorded_expt1(
            recorded_pairs_info, scenes, scene_conds, config,
            split_trials_path, init_sound_number=catch_trial_idx
            )

        # Create trials for model audio, scene_conds == 1
        model_trial_info, model_pairs_info = trials_util.model_expt1(
            expt_name, sound_group, scenes, scene_conds,
            config, recorded_trial_idx, split_trials_path
            )
        model_pairs_info.to_csv(os.path.join(
            full_experiment_path, f'model_pairs_split{split:02d}_info.csv'
            ))
        if config["n_catch_trials"] > 0:
            all_trial_info = pd.concat(
                [catch_trial_info, recorded_trial_info, model_trial_info]
                )
        else:
            all_trial_info = pd.concat([recorded_trial_info, model_trial_info])
        all_trial_info.to_csv(os.path.join(
            full_experiment_path, f'trial_split{split:02d}_info.csv'
            ))
        with open(os.path.join(trials_path, f"split{split:02d}_conditions.json"),"w") as outfile:
            # 0 is everyday audio, 1 is model
            split_conditions = ((all_trial_info["condition"] == "model")*1).tolist()
            json.dump(split_conditions, outfile)

    # Copy catch trials to both split folders
    if config["n_catch_trials"] > 0:
        print("Adding catch trials to split folders!")
        wavs_in_catch_trials = glob(os.path.join(
            full_catch_trial_path, "trials", "*.wav"
            ))
        for w in wavs_in_catch_trials:
            for split in range(2):
                split_trials_path = os.path.join(trials_path, f"split{split:02d}", "")
                copyfile(w, os.path.join(
                    split_trials_path, os.path.basename(w)
                    ))

    # Amplitude normalize sounds
    # 1) Find the max amplitude sound
    print("Amplitude normalizing the sounds...")
    ms = []
    for split in range(2):
        split_trials_path = os.path.join(trials_path, f"split{split:02d}", "")
        wavfns = glob(os.path.join(split_trials_path, "*.wav"))
        for wfn in wavfns:
            x, sr = sf.read(wfn)
            ms.append(np.max(np.abs(x)))
    mnorm = np.max(ms) * 1.001
    # 2) Normalize such that the max amplitude
    # sound is set to ~1 (with some space not to clip)
    for split in range(2):
        split_trials_path = trials_path + f"/split{split:02d}/"
        wavfns = glob(split_trials_path + "*.wav")
        for wfn in wavfns:
            x, sr = sf.read(wfn)
            sf.write(wfn, x/mnorm, sr)

    print("Complete!")
    print("Check out experiment at: ", full_experiment_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expt-name", help="set of inferences")
    parser.add_argument("sound-group", help="sounds used in inferences")
    parser.add_argument("--n-foils-per-source", type=int, default=85,
                        help="number of foils to gather for each source. could be \
                        used by different experiment participants \
                        (how we use it), or in an >2AFC task."
                        )
    parser.add_argument("--max-sources-per-scene", type=int, default=4,
                        help="number of sources to include \
                              from each sound mixture."
                        )
    parser.add_argument("--n-catch-trials", type=int, default=10,
                        help="extra trials to assess whether \
                              participants are guessing"
                        )
    args = parser.parse_args()
    config = {
        "n_foils_per_source": args.n_foils_per_source,
        "max_sources_per_scene": args.max_sources_per_scene,
        "n_catch_trials": args.n_catch_trials,
        # easy to not repeat because there's a large database of FUSS sounds
        "unique_recorded_foils": True
    }
    main(args.expt_name, args.sound_group, config)
