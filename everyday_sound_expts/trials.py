import os
import json
import random
from shutil import copyfile

import jams
import torch
import numpy as np
import pandas as pd
import soundfile as sf

import inference.io
from renderer.util import ramp_sound, spectrum_from_torch_rfft
import everyday_sound_expts.fuss_util as fuss_util
import everyday_sound_expts.select_mixtures
from util.sample import sample_delta, manual_seed
from util.context import context

max_foils_per_recording = 10

####################
#
# Experiment 1
#
#####################


def recorded_foils_expt1(mixture_info, expt_folder, sound_group, config):
    """ Experiment 1: get "foil" options for 2AFC experiment,
        specifically for the recorded sound trials

    Foil sources should be:
        Metadata
        - (a) From the same dataset as target
              (eg. FUSS if FUSS, match background, etc.)
        - (b) Should not be present in any of the scenes included
        - (c) Should not be in any of the excluded sound classes
        Sound edits
        - (d) Should be trimmed/padded to have similar onset to target sound
        - (e) Should be normalized to have the same maximum amplitude
              as the target sound

    Table describing the pairs:
    table_columns = ["dataset_name",
                        "exemplar" -- see select_mixtures.clip_sounds_and_log
                        "scene_dest": in the experiment folder (see same),
                        "source_dest" in the experiment_folder (see same),
                        "source_category", "bg_or_fg", "sr" -- see same
    for foil_idx in range(config["n_foils_per_source"]):
        fields = ["foil{}_src": origin of foil wavfile
                  "...dest", where the foil is saved in the experiment folder
                "...exemplar", "...source_idx", "...category"]

    """

    # collect experiment trial info in this table T
    T = []
    table_columns = [
        "dataset_name", "exemplar", "scene_dest",
        "source_dest", "source_category", "bg_or_fg", "sr"
        ]
    for foil_idx in range(min(config["n_foils_per_source"], max_foils_per_recording)):
        fields = ["src", "dest", "exemplar", "source_idx", "category"]
        fields = [f"foil{foil_idx:02d}_" + f for f in fields]
        table_columns.extend(fields)

    # (b) Get list of all the source files in all
    # of the scenes - do not use a foil which is in this list.
    do_not_use = []
    mixture_info = mixture_info.reset_index()
    for d in mixture_info.iterrows():
        row_number = d[0]
        jams_src = mixture_info["jams_src"][row_number]
        metadata = jams.load(jams_src)
        for source_idx in range(config["n_sounds"]):
            source_metadata = metadata.annotations[0].data[source_idx]
            do_not_use.append(source_metadata.value["source_file"])
    # Do not use the same foil for a single participant twice
    participant_do_not_use = {
        i: [] for i in range(config["n_foils_per_source"])
        }

    for d in mixture_info.iterrows():
        print("now on ", d[0])
        row_number = d[0]
        dataset_name = mixture_info["dataset_name"][row_number]
        jams_src = mixture_info["jams_src"][row_number]
        metadata = jams.load(jams_src)

        _, n_datapoints, naming = fuss_util.get_paths(sound_group, config)
        naming["foil_dest"] = lambda i, source_idx, foil_idx: os.path.join(
            os.environ["sound_dir"],
            sound_group,
            "experiments",
            expt_folder,
            "recorded_foils",
            f"FUSS-train_{i:05d}_source{source_idx:02d}_foil{foil_idx:02d}.wav"
        )

        for source_idx in range(config["n_sounds"]):
            print("source idx:", source_idx)

            # Check if the sound is silent
            sound_to_check, sr = sf.read(
                mixture_info[f'source{source_idx:02d}_dest'][row_number]
                )
            if np.all(sound_to_check == 0):
                print(
                    "Skipping because silent:",
                    mixture_info[f'source{source_idx:02d}_dest'][row_number]
                    )
                continue

            print("Looking for foils for ",
                  mixture_info[f'source{source_idx:02d}_dest'][row_number]
                  )
            source_metadata = metadata.annotations[0].data[source_idx]

            T_entry = [
                dataset_name,
                mixture_info["exemplar"][row_number],
                mixture_info['scene_dest'][row_number],
                mixture_info[f'source{source_idx:02d}_dest'][row_number],
                mixture_info[f'source{source_idx:02d}_category'][row_number],
                "bg" if source_idx == 0 else "fg",
                mixture_info["sr"][row_number]
                ]
            start_time_to_match = mixture_info[f'source{source_idx:02d}_t0'][row_number]

            n_foils_for_this_source = 0
            while n_foils_for_this_source < min(config["n_foils_per_source"], max_foils_per_recording):
                print("trying foil...")

                # (a) Load potential target from same dataset
                i = np.random.randint(n_datapoints)
                foil_metadata = naming["metadata"](i)
                fm = jams.load(foil_metadata)

                # (a) From the same dataset as target
                #     and background if background
                if source_metadata.value["role"] == "background":
                    # Only look at first (which should be background)
                    annotation_idxs = range(1)
                else:
                    annotation_idxs = list(range(
                        1, len(fm.annotations[0].data)
                        ))
                    random.shuffle(annotation_idxs)

                success, foil_category = fuss_util.check_for_categories(
                    fm, annotation_idxs, config["categories_to_exclude"][dataset_name]
                    )
                if not success:
                    continue
                for aidx in annotation_idxs:

                    # (b) Check if present in any target scenes
                    already_in_set_1 = fm.annotations[0].data[aidx].value["source_file"] in do_not_use
                    already_in_set_2 = fm.annotations[0].data[aidx].value["source_file"] in participant_do_not_use[n_foils_for_this_source]
                    if already_in_set_1 or already_in_set_2:
                        print(f"Not using because already in set 1: \
                              {already_in_set_1} or set 2: {already_in_set_2}")
                        continue

                    # (c) Should not be in any of the excluded sound classes 
                    for excluded_category in config["categories_to_exclude"][dataset_name]:
                        if excluded_category in foil_category[aidx]:
                            continue

                    # If we got this far, this foil is a keeper
                    # Need to create the sound and fill these foil categories
                    # for the table:
                    # ["src", "to_load", "exemplar", "source_idx", "category"]
                    # (d) Should be trimmed/padded to have similar onset
                    #     to target sound
                    foil_onset = fm.annotations[0].data[aidx].time
                    fn_to_load = naming["source_src"](i, aidx, fm)
                    x_foil, sr_foil = sf.read(fn_to_load)
                    if sr_foil != mixture_info["sr"][row_number]:
                        raise Exception("Sampling rates are not equal: are \
                                        you using different datasets for foil \
                                        and target?")
                    if foil_onset < start_time_to_match:
                        pad_before_in_sec = start_time_to_match - foil_onset
                        t1 = config["clip_duration"] - pad_before_in_sec
                        x_foil = np.concatenate([
                            np.zeros(int(pad_before_in_sec*sr_foil)),
                            x_foil[:int(sr_foil*t1)]
                            ])
                    elif foil_onset > start_time_to_match:
                        start_point = foil_onset - start_time_to_match
                        t1 = start_point + config["clip_duration"]
                        x_foil = x_foil[int(sr_foil*start_point):int(sr_foil*t1)]
                    else:
                        t0 = foil_onset
                        t1 = foil_onset + config["clip_duration"]
                        x_foil = x_foil[int(sr_foil*t0):int(t1*sr_foil)]

                    # (e) Should be normalized to have the same maximum
                    # amplitude as the target sound
                    x_foil = ramp_sound(x_foil, sr_foil, ramp_duration=0.005)
                    x_foil = x_foil - x_foil.mean()
                    x_max = np.max(np.abs(x_foil))
                    # If the foil is silence don't include it
                    if x_max == 0:
                        continue
                    x_foil = x_foil/x_max
                    x_source, _ = sf.read(
                        mixture_info[f'source{source_idx:02d}_dest'][row_number]
                        )
                    x_foil *= np.max(np.abs(x_source))

                    # Save it!
                    fn_to_save = naming["foil_dest"](
                        mixture_info["exemplar"][row_number], source_idx,
                        n_foils_for_this_source
                        )
                    sf.write(fn_to_save, x_foil, sr_foil)
                    print("Found foil! ", [
                        fn_to_load, i, aidx, foil_category[aidx]
                        ])
                    T_entry.extend([
                        fn_to_load, fn_to_save, i, aidx, foil_category[aidx]
                        ])
                    if config["unique_recorded_foils"]:
                        participant_do_not_use[n_foils_for_this_source].append(
                            fm.annotations[0].data[aidx].value["source_file"]
                            )

                    n_foils_for_this_source += 1
                    break

            T.append(tuple(T_entry))

    print("Done collecting recorded foils.")
    D = pd.DataFrame.from_records(T, columns=table_columns)

    return D


def recorded_expt1(pairs_info, scenes, scene_conds, config, trials_path, init_sound_number=None, catch_trial=False):
    """ Randomly select scene-source-foil sets for
        the everyday sound trials of experiment one
    """

    # Keep track of experiment info
    T = []
    table_columns = [
        "scene_idx", "sound_number", "condition",
        "scene_dest", "scene_wav_to_play", "source_dest",
        "source_wav_to_play", "source_idx"
        ]
    n_foils_per_source = config["n_foils_per_source"]
    max_sources_per_scene = config["max_sources_per_scene"]
    for foil_idx in range(n_foils_per_source):
        table_columns.extend([
            f"foil{foil_idx:02d}_dest",
            f"foil{foil_idx:02d}_wav_to_play"
            ])

    print("Saving recorded foils")
    sound_number = 0 if init_sound_number is None else init_sound_number
    for scene_idx, scene in enumerate(scenes):
        print(f"Scene: {scene_idx}")

        # Skip if it's a model trial
        if scene_conds[scene_idx] == 1:
            continue

        # Select one of the sources in the scene
        pairs_with_this_scene = pairs_info[pairs_info["scene_dest"] == scene]
        n_possible_sources = len(pairs_with_this_scene)
        source_idxs = np.random.choice(
            range(n_possible_sources),
            size=min(n_possible_sources, max_sources_per_scene),
            replace=False
            )
        for source_idx in source_idxs:
            this_trial = pairs_with_this_scene.iloc[source_idx]

            # Copy sound files into the trial folder
            T_entry = [scene_idx, sound_number, "record"]
            scene_path = os.path.join(
                trials_path, f"scene{sound_number:03d}.wav"
                )
            T_entry.extend([this_trial["scene_dest"], scene_path])
            copyfile(this_trial["scene_dest"], scene_path)
            source_path = os.path.join(
                trials_path, f"source{sound_number:03d}.wav"
                )
            T_entry.extend([this_trial["source_dest"], source_path, source_idx])
            copyfile(this_trial["source_dest"], source_path)
            # Use all the foils -- if you want a different experiment
            # with different XAFC, run again with different n_foils_per_source
            for foil_idx in range(config["n_foils_per_source"]):
                # Need all catch trials to be the same across all participants!
                dest_idx = (foil_idx % max_foils_per_recording) if not catch_trial else 0
                foil_path = os.path.join(
                    trials_path, f"foil{sound_number:03d}-{foil_idx:02d}.wav"
                    )
                T_entry.extend([
                    this_trial[f"foil{dest_idx:02d}_dest"], foil_path
                    ])
                copyfile(this_trial[f"foil{dest_idx:02d}_dest"], foil_path)
            T.append(T_entry)
            sound_number += 1

    D = pd.DataFrame.from_records(T, columns=table_columns)
    return D, sound_number


def model_expt1(expt_name, sound_group, scenes, scene_conds, config, sound_number, trials_path):
    """
        For experiment one, loads in sources inferred from
        the model as well as foils for them.
        Requirements of foils:
            Foils should come from scenes which are not included
            in the model inference

        Returns:
            trial_info: includes the basic information needed to run each
                        experiment trial
            model_pairs_info: includes more specific information about the
                              specific paths where we retrieved the source info
    """

    # Keep track of experiment info -- add to trial_info from recorded_trials!
    T_trial = []
    trial_table_columns = [
        "scene_idx", "sound_number", "condition",
        "scene_dest", "scene_wav_to_play", "source_dest",
        "source_wav_to_play", "source_idx"
        ]
    n_foils_per_source = config["n_foils_per_source"]
    max_sources_per_scene = config["max_sources_per_scene"]
    for foil_idx in range(n_foils_per_source):
        trial_table_columns.extend([
            f"foil{foil_idx:02d}_dest", f"foil{foil_idx:02d}_wav_to_play"
            ])

    # Keep track of more detailed model info -- save on its own
    T_model = []
    model_table_columns = [
        "sound_name", "source_src", "source_idx",
        "source_category", "bg_or_fg", "sr"
        ]
    for foil_idx in range(n_foils_per_source):
        foil_table_columns = [
            "sound_name", "src", "source_idx",
            "source_category", "bg_or_fg", "sr"
            ]
        model_table_columns.extend([
            f"foil{foil_idx:02d}_{ft}" for ft in foil_table_columns
            ])

    # Load in all possible sounds to use first
    error_msgs = []
    all_foil_entries = []
    all_foil_waves = []
    all_T_trial_entries = []
    all_T_model_entries = []
    for scene_idx, scene in enumerate(scenes):
        print(
            f"We're on scene {scene_idx}, ",
            scene, " include as model trial?",
            scene_conds[scene_idx] == 1
            )

        # Load in torch ckpt and get source_wave and its info
        try:
            T_entries, source_waves, ckpt_errors = get_sources_from_ckpt(
                scene, expt_name, sound_group, max_sources_per_scene
                )
            error_msgs.extend(ckpt_errors)
        except FileNotFoundError as e:
            print(str(e))
            print(f"Couldn't find {scene}!")
            error_msgs.append(f'FileNotFoundError,{scene}\n')
            continue

        if scene_conds[scene_idx] == 0:
            all_foil_entries.extend(T_entries)
            all_foil_waves.extend(source_waves)

        elif scene_conds[scene_idx] == 1:
            T_model_entries = T_entries
            all_T_model_entries.extend(T_model_entries)
            for T_model_entry, source_wave in zip(T_model_entries, source_waves):
                # Copy/write sound files into the trial folder
                T_trial_entry = [scene_idx, sound_number, "model"]
                trial_scene_path = os.path.join(
                    trials_path, f"scene{sound_number:03d}.wav"
                    )
                T_trial_entry.extend([scene, trial_scene_path])
                copyfile(scene, trial_scene_path)
                source_path = os.path.join(
                    trials_path, f"source{sound_number:03d}.wav"
                    )
                T_trial_entry.extend([
                    T_model_entry[1], source_path, T_model_entry[2]
                    ])
                sf.write(source_path, source_wave, T_model_entry[-1])
                all_T_trial_entries.append(T_trial_entry)
                sound_number += 1

    # Sample from all_foil_entries for each trial in all_T_trial_entries
    n_foils_to_choose_from = len(all_foil_entries)
    n_trials_to_choose_for = len(all_T_trial_entries)
    for participant_idx in range(config["n_foils_per_source"]):
        # Ideally, one participant would never encounter a foil more than once
        # But since there may be an uneven number of model sources across
        # splits, it may be necessary to repeat a few
        # On the other hand, it is okay if there are a few repeats for the
        # same model-source-sound across participants as long as its randomized

        if n_foils_to_choose_from >= n_trials_to_choose_for:
            foil_idxs_to_use = np.random.permutation(
                np.arange(n_foils_to_choose_from)
                )[:n_trials_to_choose_for]
        else:
            foil_idxs_to_use = []
            while len(foil_idxs_to_use) < n_trials_to_choose_for:
                foil_idxs_to_use.extend(list(np.random.permutation(
                    np.arange(n_foils_to_choose_from)
                    )[:n_trials_to_choose_for-len(foil_idxs_to_use)]))

        assert len(foil_idxs_to_use) == n_trials_to_choose_for
        for trial_idx, (T_trial_entry, T_model_entry) in enumerate(zip(all_T_trial_entries, all_T_model_entries)):
            T_entry_foil = all_foil_entries[foil_idxs_to_use[trial_idx]]
            foil_wave = all_foil_waves[foil_idxs_to_use[trial_idx]]
            foil_path = os.path.join(
                trials_path, f"foil{T_trial_entry[1]:03d}-{participant_idx:02d}.wav"
                )
            T_trial_entry.extend([T_entry_foil[1], foil_path])
            sf.write(foil_path, foil_wave, T_entry_foil[-1])
            T_model_entry.extend(T_entry_foil)

    T_trial.extend(all_T_trial_entries)
    T_model.extend(all_T_model_entries)
    D_trial = pd.DataFrame.from_records(T_trial, columns=trial_table_columns)
    D_model = pd.DataFrame.from_records(T_model, columns=model_table_columns)

    error_log_fn = os.path.join(
        *trials_path.split(os.sep)[:-2], 'errorlog.csv'
        )
    print("Error log at ", error_log_fn)
    write_mode = "a" if os.path.isfile(error_log_fn) else "w"
    error_log = open(error_log_fn, write_mode)
    for error_msg in np.unique(error_msgs):
        error_log.write(error_msg)
    error_log.close()

    return D_trial, D_model


def catch_expt1(config, mixture_info, experiment_folder):
    """ Select a small number of natural sounds that will
        be added to the trials to exclude participants
        if they're just guessing
    """

    catch_trial_config = config.copy()
    # +1 so it will generate different sounds to the initial
    catch_trial_config["seed"] = catch_trial_config["seed"] + 1
    manual_seed(catch_trial_config["seed"])
    catch_trial_config["sound_group"] = experiment_folder
    # Generate enough so that if there are replicates
    # with mixture_info we can exclude them
    catch_trial_config["n_samples"]["fuss"] = config["n_catch_trials"]
    dataset_name, n_datapoints, naming = fuss_util.get_paths(
        catch_trial_config["sound_group"], catch_trial_config
        )
    catch_trial_dict = everyday_sound_expts.select_mixtures.clip_sounds_and_log(
        dataset_name, n_datapoints, naming, catch_trial_config,
        exemplars_to_exclude=mixture_info["exemplar"]
        )
    # For each catch_trial scene, only have 1 trial
    catch_trial_config["max_sources_per_scene"] = 1

    return catch_trial_dict, catch_trial_config

################
#
# Experiment 2
#
#################


terminal_colors = {
    "RED": "\033[91m",
    "ENDC": "\033[0m"
    }


def format_expt2(expt_name, sound_group, scenes, mixture_info, expt1_sounds_to_exclude, config, trials_path, mode, sound_number=0):
    """ Create experiment 2 trial tables (see Fig8E or example on website)
        Returns a dataframe describing the experiment trials

        The columns in the dataframe are:
            sound_number: indexing for saving the relevant waves
            trial_type:
            - full: all recognizable model sounds on rows,
                    premixture sounds on columns
            - catch: premixture sounds on rows and columns
            exemplar: the FUSS number of the scene that is the focus
            n_recordings: how many recordings in the columns
            rec_source_idxs: numbering of sources in the original recordings
                             (eg source00, source01) in the order they are
                             placed in columns
            col_src: csv listing where the sounds for the recordings come from
                     (columns in expt setup)
            col_dest: csv listing where the recordings are saved to show
                      up in the experiment (columns in expt setup)
            n_rowsounds: how many sounds in the rows (whether they are
                         model sounds or catch sounds)
            model_row_idxs: which row sounds are matching model sounds
            foil_row_idxs: which row sounds are non-matching model sounds
                           (having both of these is useful in the "partial"
                            trial type, if we use it)
            row_src: csv listing where the row sounds come from
            row_dest: csv listing where the row sounds saved to
                      show up in the experiment
    """
    table_columns = [
        "sound_number", "trial_type", "exemplar", "n_recordings",
        "rec_source_idxs", "col_src", "col_dest", "n_rowsounds",
        "model_row_idxs", "foil_row_idxs", "row_src", "row_dest"
        ]

    T_entries = []
    error_msgs = []
    n_skipped_full_scenes = 0
    n_skipped_for_silence = 0
    n_skipped_for_unrecognizable = 0
    for scene_idx, scene in enumerate(scenes):

        # Column sounds: premixture recordings
        row_number = np.where(mixture_info["scene_dest"] == scene)[0][0]
        # Save the scene for this trial
        this_trial = mixture_info.iloc[row_number]
        copyfile(this_trial["scene_dest"], trials_path + f"scene{sound_number:03d}.wav")
        T_entry = [sound_number, mode, this_trial["exemplar"]]
        # Save recordings which are audible and recognizable for the
        # columns of this trial
        # Don't randomize the order here because it will have to happen
        # uniquely for each participant
        n_recordings_in_trial = 0
        r = range(config["n_sounds"])
        col_src = []
        col_dest = []
        rec_source_idxs = []
        for source_idx in r:
            if mixture_info[f'source{source_idx:02d}_dest'][row_number] in expt1_sounds_to_exclude["recorded"]:
                n_skipped_for_unrecognizable += 1
                print(
                    "Skipping because unrecognizable: ",
                    mixture_info[f'source{source_idx:02d}_dest'][row_number]
                    )
                continue
            # Check if the sound is silent
            rec_wave, _ = sf.read(
                mixture_info[f'source{source_idx:02d}_dest'][row_number]
                )
            if np.all(rec_wave == 0):
                n_skipped_for_silence += 1
                print(
                    "Skipping because silent:",
                    mixture_info[f'source{source_idx:02d}_dest'][row_number]
                    )
                continue
            # Save for sound number
            col_src.append(this_trial[f"source{source_idx:02d}_dest"])
            col_dest.append(os.path.join(
                trials_path,
                f"col{sound_number:03d}_{n_recordings_in_trial:02d}.wav"
                ))
            rec_source_idxs.append(source_idx)
            copyfile(
                this_trial[f"source{source_idx:02d}_dest"],
                os.path.join(
                    trials_path,
                    f"col{sound_number:03d}_{n_recordings_in_trial:02d}.wav"
                ))
            n_recordings_in_trial += 1
        T_entry.extend([
            n_recordings_in_trial, rec_source_idxs, col_src, col_dest
            ])

        # Row sounds: model inferences if mode == "full"
        # or the same premixture sounds if mode == "catch"
        if mode == "full":
            # Model: Load in torch ckpt and get source_wave and its info
            row_scene = scene
            try:
                T_model_entries, model_waves, ckpt_errors = get_sources_from_ckpt(
                    row_scene, expt_name, sound_group, 4
                    )
                error_msgs.extend(ckpt_errors)
            except FileNotFoundError as e:
                print(str(e))
                print(f"Couldn't find {row_scene}!")
                error_msgs.append(f'FileNotFoundError,{row_scene}\n')
                continue
            # Check if model inferences are recognizable and save them if so.
            r = range(len(model_waves))
            model_row_idxs = []
            row_src = []
            row_dest = []
            n_rowsounds_in_trial = 0
            for r_idx in r:
                T_model_entry = T_model_entries[r_idx]
                model_wave = model_waves[r_idx]
                if (T_model_entry[1], T_model_entry[2]) in expt1_sounds_to_exclude["model"]:
                    print(
                        "Skipping because unrecognizable, ",
                        (T_model_entry[1], T_model_entry[2])
                        )
                    continue
                row_src.append(T_model_entry[1])
                source_path = os.path.join(
                    trials_path,
                    f"row{sound_number:03d}_{n_rowsounds_in_trial:02d}.wav"
                    )
                row_dest.append(source_path)
                model_row_idxs.append(r_idx)
                sf.write(source_path, model_wave, T_model_entry[-1])
                n_rowsounds_in_trial += 1
            if n_rowsounds_in_trial == 0:
                n_skipped_full_scenes += 1
                print("Skipping full scene because all \
                      its sources unrecognizable: ", scene)
                continue
            T_entry.extend([
                n_rowsounds_in_trial, model_row_idxs, "", row_src, row_dest
                ])

        elif mode == "catch":
            # Simply copy above recordings above
            if scene_idx == 0:
                n_catch_trials = 0
            row_src = []
            row_dest = []
            source_idxs = np.arange(n_recordings_in_trial)
            for c_idx, source_idx in enumerate(source_idxs):
                foil_path = os.path.join(
                    trials_path, f"row{sound_number:03d}_{c_idx:02d}.wav"
                )
                row_dest.append(foil_path)
                row_src.append(col_src[source_idx])
                copyfile(col_src[source_idx], foil_path)
            n_rowsounds = n_recordings_in_trial
            ridxs = rec_source_idxs
            n_catch_trials += 1
            T_entry.extend([n_rowsounds, "", ridxs, row_src, row_dest])

        T_entries.append(T_entry)
        sound_number += 1

    if mode == "full":
        print(terminal_colors["RED"] +
              f"Skipped {n_skipped_full_scenes} full scenes." +
              terminal_colors["ENDC"]
              )
        print(terminal_colors["RED"] +
              f"Skipped {n_skipped_for_silence} sources for silence." +
              terminal_colors["ENDC"]
              )
        print(terminal_colors["RED"] +
              f"Skipped {n_skipped_for_unrecognizable} \
                sources for unrecognizable." +
              terminal_colors["ENDC"]
              )
    elif mode == "catch":
        print(
            terminal_colors["RED"] +
            f"Created {n_catch_trials} total catch_trials." +
            terminal_colors["ENDC"]
            )
    D = pd.DataFrame.from_records(T_entries, columns=table_columns)

    return D, sound_number


def catch_expt2(config, mixture_info, experiment_folder):
    """ Select a small number of natural sounds that will be
        added to the trials to exclude participants if they're just guessing
    """

    catch_trial_config = config.copy()
    # +1 so it will generate different sounds to the initial
    catch_trial_config["seed"] = catch_trial_config["seed"] + 1
    manual_seed(catch_trial_config["seed"])
    catch_trial_config["sound_group"] = experiment_folder
    # Generate enough so that if there are replicates with
    # mixture_info we can exclude them
    catch_trial_config["n_samples"]["fuss"] = config["n_catch_trials"]
    dataset_name, n_datapoints, naming = fuss_util.get_paths(
        catch_trial_config["sound_group"], catch_trial_config
        )
    catch_trial_dict = everyday_sound_expts.select_mixtures.clip_sounds_and_log(
        dataset_name, n_datapoints, naming,
        catch_trial_config, exemplars_to_exclude=mixture_info["exemplar"]
        )

    return catch_trial_dict, catch_trial_config

#################
#
# Load in sounds from a model
#
#################


def get_sources_from_ckpt(scene_path, expt_name, sound_group, max_sources_per_scene):
    """
    The experiments with everyday sounds require sounds
    rendered from the model's inferred sources
    This function loads in checkpoint file, samples the MAP
    source from the approximate posterior, and returns the audible sources
    """
    with torch.no_grad():
        # Load scene from inference checkpoint
        sound_name = os.path.splitext(os.path.basename(scene_path))[0]
        obs, sr = sf.read(os.path.join(
            os.environ["sound_dir"], sound_group, sound_name + ".wav"
            ))
        desired_duration = len(obs)/(1.*sr)
        results_file = os.path.join(
            os.environ["inference_dir"], "results.json"
            )
        with open(results_file, "r") as f:
            result = json.load(f)
        expt_to_keep = result["keep"][-1][0]
        round_idx = len(result["keep"]) - 1
        source_src = os.path.join(
            os.environ["inference_dir"],
            f"round{round_idx:03d}-{expt_to_keep:03d}", ""
            )
        scene, _ = inference.io.restore_hypothesis(source_src)
        config = inference.io.get_config(source_src)

        # Sample a scene description
        with context(config, batch_size=1):
            # sample_delta corresponds to choosing MAP
            # this will smooth over epsilon noise in the sampled GPs
            scene.pq_forward(sample=sample_delta)

        # Incorrect duration is an indication that
        # something has gone wrong for inference
        if not correct_duration(scene.scene_wave[0, :].cpu().numpy(), scene.audio_sr, desired_duration):
            return [], [], [f'TooShort,{scene_path}\n']

        # Permute sources
        n_possible_sources = scene.n_sources.item()
        source_idxs = np.random.permutation(range(n_possible_sources))

        # Collect information on each inferred source and check audibility
        T_model_entries = []
        source_waves = []
        error_msgs = []
        for source_idx in source_idxs:
            source_wave = scene.sources[source_idx].source_wave[0, :].cpu().numpy()
            if not audible(source_wave, scene.audio_sr, config["renderer"]["tf"]["rms_ref"]):
                error_msgs.append(
                    f'NotAudible,{sound_name}_source{source_idx:02d}\n'
                    )
                continue
            source_waves.append(source_wave)
            source_type = scene.sources[source_idx].source_type
            T_model_entry = [
                sound_name, source_src, source_idx, source_type,
                "n/a", scene.audio_sr
                ]
            T_model_entries.append(T_model_entry)
            if len(source_waves) == max_sources_per_scene:
                break

        return T_model_entries, source_waves, error_msgs


def audible(s, sr, rms_ref):
    """
    Check (imperfectly) if the sound will be audible
    on headphones with a heuristic
    If there is energy above 25 dB above 100 Hz, consider it audible
    If there isn't, check whether there is energy above 60 dB under 100 Hz.
    """
    f, spectrum = spectrum_from_torch_rfft(torch.Tensor(s), sr, rms_ref)
    audible = torch.any(spectrum[f > 100] >= 25)
    if not audible:
        audible = torch.any(spectrum[f <= 100] > 60)
    return audible


def correct_duration(s, sr, desired_duration):
    """ Check if the sound is shorter than a desired duration """
    too_short = len(s)/(1.*sr) < desired_duration
    return (not too_short)
