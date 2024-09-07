import os
import yaml
import argparse

import numpy as np
import jams
import soundfile as sf
import pandas as pd

import everyday_sound_expts.fuss_util as fuss_util
from util.sample import manual_seed


def main(config):

    # Make the new folder if it doesn't exist yet
    manual_seed(config["seed"])
    audio_folder = os.path.join(os.environ["sound_dir"], config["sound_group"], "")
    if not os.path.isdir(audio_folder):
        os.mkdir(audio_folder)
    # Log the config file
    with open(os.path.join(audio_folder, "config.yaml"), 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

    # Get sounds and save
    dataset_name, n_datapoints, naming = fuss_util.get_paths(
        config["sound_group"], config
        )
    mixture_info = clip_sounds_and_log(
        dataset_name, n_datapoints, naming, config
        )
    mixture_info.to_csv(os.path.join(audio_folder, 'mixture_info.csv'))


def clip_sounds_and_log(dataset_name, n_datapoints, naming, config, exemplars_to_exclude=[]):
    """ Get scenes and premixture sound to use in model inference
        and then human experiments, record a log of which sounds
        were selected and their properties.

        Scenes/sources should:
            - Require some interval of only background before
              first foreground event begins
            - Have a required number and duration of premixture
              recordings, in the right categories

        Create a table describing the sounds:
        columns = ["dataset_name": fuss or desed or etc.
                "exemplar": number indexing the datapoint in the dataset
                "scene_src": path to original mixture file
                "scene_dest": where the mixture sound is saved for
                                experiment/inference
                "jams_src": path to the original annotations file
                "t0": time in the original clip where the experiment clip
                        begins (sec)
                "t1": time in the original clip where the experiment clip
                        ends (sec)
                "n_sources": number of sources including background
                "sr": audio sampling rate in Hz
                ]
        extra columns for sources:
            [
            "source{source_idx:02d}_idx": number of source in the mixture
            "...src": path to the original source file
            "...dest": where the source sound has been saved for the experiment
            "...t0": starting time of the source within the clip (sec)
            "...t1": ending time of the source within the clip (sec)
            "...category": class label of the source
            ]

        exemplars_to_exclude is used to make catch trials
    """

    print("~~~~ Collecting sounds from ", dataset_name)

    # Prepare pandas datatable to save all information on the chosen scenes
    table_columns = [
        "dataset_name", "exemplar", "scene_src",
        "scene_dest", "jams_src", "t0", "t1",
        "n_sources", "sr"
        ]
    for source_idx in range(config["n_sounds"]):
        fields = ["idx", "src", "dest", "t0", "t1", "category"]
        fields = [f"source{source_idx:02d}_" + f for f in fields]
        table_columns.extend(fields)

    # Loop to collect sounds
    T = []
    collected = 0
    while collected < config["n_samples"][dataset_name]:
        # Choose random file and load metadata
        i = np.random.randint(n_datapoints)
        if i in exemplars_to_exclude:
            continue
        metadata = naming["metadata"](i)
        m = jams.load(metadata)
        # Decide whether to include the sound
        # 1. Require some interval of only background before
        # first foreground event begins.
        # Find first event
        _t0 = fuss_util.get_first_foreground_onset(m)
        if _t0 is False:
            continue
        a = max(_t0 - config['max_time_before_onset'], 0.0)
        if (_t0 - a) < config['min_time_before_onset']:
            continue
        # Generate a random starting point before the first event
        t0 = np.random.uniform(low=a, high=_t0-config['min_time_before_onset'])
        t1 = t0 + config['clip_duration']
        # 2. Check that there are the required number and duration
        # of sounds in the clip of interest
        right_number_of_sounds, source_in_interval, event_timings = fuss_util.check_for_n_sounds_inside(
            m, t0, t1, config['n_sounds'], config['min_recording_duration']
            )
        if not right_number_of_sounds:
            continue
        # 3. Check that all sounds within the interval
        # come from the allowed categories
        source_idxs = np.where(source_in_interval)[0]
        correct_categories, actual_categories = fuss_util.check_for_categories(
            m, source_idxs,
            config["categories_to_exclude"][dataset_name]
            )
        if not correct_categories:
            continue
        # Save the data necessary for experiment and model inference
        T_entry = [
            dataset_name, i,
            naming["scene_src"](i),
            naming["scene_dest"](i),
            naming["metadata"](i),
            t0, t1, sum(source_in_interval)
            ]
        # 4. Save the mixture clip
        x_scene, sr, _ = fuss_util.clip_ramp_normalize(
            naming["scene_src"](i), t0, t1
            )
        T_entry.append(sr)
        # 5. Save single source clips
        found_all_sources = None
        x_sources = []
        for source_idx in source_idxs:
            source_t0, source_t1 = event_timings[source_idx]
            source_category = actual_categories[source_idx]
            try:
                x, sr, _ = fuss_util.clip_ramp_normalize(
                    naming["source_src"](i, source_idx, m),
                    t0, t1
                    )
                if np.all(x == 0):
                    print("One of the single sources is a vector of zeros. \
                          Skip this scene")
                    found_all_sources = False
                    break
                x_sources.append(x)
            except RuntimeError as e:
                if "Error opening" in str(e):
                    print(e)
                    print("Error: couldn't find: ",
                          naming["source_src"](i, source_idx, m)
                          )
                    print("Will not include this sound.")
                    found_all_sources = False
                    break
            T_entry.extend([
                source_idx,
                naming["source_src"](i, source_idx, m),
                naming["source_dest"](i, source_idx),
                source_t0, source_t1, source_category
                ])
        if found_all_sources is False:
            continue

        # Got to the end, so save all the sounds
        if collected == 0:
            os.makedirs(os.path.join(
                os.environ["sound_dir"],
                config["sound_group"],
                "single_sources",
                ""
                ))
        print("Found a sound!")
        print("Event timings: ", [event_timings[si] for si in source_idxs])
        print("Categories: ", [actual_categories[si] for si in source_idxs])
        sf.write(naming["scene_dest"](i), x_scene, sr)
        for e_idx, source_idx in enumerate(source_idxs):
            sf.write(
                naming["source_dest"](i, source_idx), x_sources[e_idx], sr
                )
        T.append(tuple(T_entry))
        collected += 1

    D = pd.DataFrame.from_records(T, columns=table_columns)
    print("~~~~ Done collecting sounds from ", dataset_name)

    return D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sound-group",
                        help="audio folder for saving")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-duration", type=float, default=2.0,
                        help="duration of mixture in seconds"
                        )
    parser.add_argument("--max-time-before-onset", type=float, default=0.100,
                        help="latest possible onset time \
                        for first premixture sound"
                        )
    parser.add_argument("--min-time-before-onset", type=float, default=0.050,
                        help="earliest possible onset time \
                            for first premixture sound")
    parser.add_argument("--n-sounds", type=int, default=3,
                        help="number of premixture sounds \
                            in scene, including background")
    parser.add_argument("--min-recording-duration", type=float, default=0.2,
                        help="minimum duration of premixture sound \
                            that counts as being in mixture, in seconds")
    parser.add_argument("--datasets", type=str, default="fuss",
                        help="which dataset to select mixtures from")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="number of scenes to select")
    args = parser.parse_args()

    config = {
        "seed": args.seed,
        "sound_group": args.sound_group,
        "clip_duration": args.clip_duration,
        "max_time_before_onset": args.max_time_before_onset,
        "min_time_before_onset": args.min_time_before_onset,
        "n_sounds": args.n_sounds,
        "min_recording_duration": args.min_recording_duration,
        "datasets": args.datasets.split(",")
        }
    config["n_samples"] = {'fuss': args.n_samples}
    config["categories_to_exclude"] = {
        "fuss": [
            "Speech",
            "Mechanisms",
            "Scratching_(performance_technique)",
            "Human_group_actions"
            ]
        }

    main(config)
