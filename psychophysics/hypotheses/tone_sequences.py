import os
import yaml
import numpy as np

import psychophysics.hypotheses.hutil as hutil
import psychophysics.generation.tone_sequences as gen
from renderer.util import freq_to_ERB
from util import context


def full_design(sound_group, config_name, overwrite={}):

    with open(os.path.join(os.environ["config_dir"], config_name + '.yaml'), 'r') as f:
        hypothesis_config = yaml.load(f, Loader=yaml.FullLoader)

    with context(
        audio_sr=hypothesis_config["renderer"]["steps"]["audio_sr"],
        rms_ref=hypothesis_config["renderer"]["tf"]["rms_ref"]
    ):
        # 1. Create sounds for inference if not already made
        settings_fn = os.path.join(os.environ["sound_dir"], sound_group, "ts_expt_settings.npy")
        if not os.path.isfile(settings_fn):
            os.makedirs(os.path.join(os.environ["sound_dir"], sound_group, ""), exist_ok=True)
            gen.make_all(sound_group, overwrite)
            settings = {"audio_sr": context.audio_sr, "rms_ref": context.rms_ref}
            np.save(settings_fn, settings)
        else:
            settings = np.load(settings_fn, allow_pickle=True).item()
        if settings["audio_sr"] != context.audio_sr or settings["rms_ref"] != context.rms_ref:
            raise Exception("Conflicting config and settings for hypothesis definition.")

    experiments = {}
    if overwrite.get("bouncing", True) is not False:
        experiments.update(x_design(hypothesis_config))
    if overwrite.get("aba", True) is not False:
        experiments.update(aba_design(hypothesis_config, overwrite.get("aba", {})))
    if overwrite.get("cumulative", True) is not False:
        experiments.update(cumul_design(hypothesis_config))
    if overwrite.get("captor", True) is not False:
        experiments.update(captor_design(hypothesis_config))
    if overwrite.get("compete", True) is not False:
        experiments.update(competition_design(hypothesis_config, overwrite.get("compete", {})))

    return experiments


def create_scene_hypothesis(events, source_assignment, config):
    # Create source placeholders
    pre_sources = [[] for _ in np.unique(source_assignment)]
    # For tone sequences, defining events with relative gap/duration 
    # ensures that they do not overlap
    # Use format for event: {"gap": float, "duration": float, "f0": float, "amplitude": float}
    events = relative_to_absolute(events)
    for event_idx, event in enumerate(events):
        pre_sources[source_assignment[event_idx]].append(event)
    # Organize into hypothesis list/dict format
    scene_hypothesis = []
    for pre_source in pre_sources:
        source_hypothesis = {}
        source_hypothesis["source_type"] = "whistle"
        # Account for "inner" ramp placement
        source_hypothesis["events"] = [
            {
                "onset": event["onset"] + config["renderer"]["ramp_duration"]/2.,
                "offset": event["offset"] - config["renderer"]["ramp_duration"]/2.
            } for event in pre_source
            ]
        # Concatenate individual pure tones into GP source, generating the inducer points.
        source_hypothesis["features"] = {}
        for feature in ["f0", "amplitude"]:
            source_hypothesis["features"][feature] = {"x": [], "y": []}
            for event in pre_source:
                ts = np.arange(
                    event["onset"] + 0.001,
                    event["offset"] + config["hypothesis"]["delta_gp"]["t"],
                    config["hypothesis"]["delta_gp"]["t"]
                    )
                source_hypothesis["features"][feature]["x"].append(ts)
                source_hypothesis["features"][feature]["y"].append(np.full(ts.shape, event[feature]))
            for a in ["x", "y"]:
                source_hypothesis["features"][feature][a] = np.concatenate(source_hypothesis["features"][feature][a])
        # Combine sources into a scene hypothesis,
        # which can be passed to Scene.hypothesize
        scene_hypothesis.append(source_hypothesis)
    return scene_hypothesis

#################
# Bouncing demo   
#################


def x_hypothesis(source_assignment, config):
    fa_vals = [
        (19.06, 72.41), (9.383, 72.41), (17.45, 72.41), (10.997, 72.41),
        (15.84, 72.41), (12.07, 72.41), (13.687, 72.41), (13.687, 72.41),
        (12.07, 72.41), (15.84, 72.41), (10.997, 72.41), (17.45, 72.41),
        (9.383, 72.41), (19.06, 72.41)
        ]
    events = []
    for event_idx, fa_val in enumerate(fa_vals):
        event = {}
        event["f0"] = fa_val[0]
        event["amplitude"] = fa_val[1]
        event["gap"] = 0.0 if event_idx > 0 else 0.054
        event["duration"] = 0.100
        events.append(event)
    return create_scene_hypothesis(events, source_assignment, config)


def x_design(config):

    Hs = {
        "crossing": [0, 1, 0,  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "bouncing": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        }

    experiments = {}
    observation_name = "Track17-01-rs-short"
    for k, v in Hs.items():
        experiments.update(
            hutil.format(x_hypothesis(v, config), observation_name, k)
            )

    return experiments

#################
# ABA demos
#################


def aba_hypothesis(source_assignment_chunk, semitone_step, onset_to_onset, n_reps, config, initial_gap=None, A_Hz=None, dB=None):

    source_assignment = np.tile(source_assignment_chunk, n_reps)

    A_Hz = 1000 if A_Hz is None else A_Hz
    B_Hz = A_Hz*(2.**(semitone_step/12.))

    A_ERB = freq_to_ERB(np.array(A_Hz)).item()
    B_ERB = freq_to_ERB(np.array(B_Hz)).item()
    dB = 70 if dB is None else dB
    tone_duration = 0.050

    events = []
    for r in range(n_reps):
        for tone_idx, f0 in enumerate([A_ERB, B_ERB, A_ERB]):
            event = {}
            event["f0"] = f0
            event["amplitude"] = dB
            if tone_idx == 0 and r == 0:
                event["gap"] = onset_to_onset/1000. - tone_duration if initial_gap is None else initial_gap
            elif tone_idx == 0 and r > 0:
                event["gap"] = 2*onset_to_onset/1000. - tone_duration
            else:
                event["gap"] = onset_to_onset/1000. - tone_duration
            event["duration"] = tone_duration
            events.append(event)

    return create_scene_hypothesis(events, source_assignment, config)


def aba_design(config, overwrite={}):
    n_rep = 4
    dfs = [1, 3, 6, 9, 12] if "df" not in overwrite.keys() else overwrite["df"]
    dts = [67, 83, 100, 117, 134, 150] if "dt" not in overwrite.keys() else overwrite["dt"]
    source_assignment_chunks = {
        "galloping": [0, 0, 0],
        "isochronous": [0, 1, 0]
        }
    experiments = {}
    for df in dfs:
        for dt in dts:
            for sk, source_assignment_chunk in source_assignment_chunks.items():
                H = aba_hypothesis(source_assignment_chunk, df, dt, n_rep, config)
                observation_name = "df{:d}_dt{:d}_rep{:d}".format(df, dt, n_rep)
                experiments.update(hutil.format(H, observation_name, sk))
    return experiments


def cumul_design(config, overwrite={}):
    n_reps = [1, 2, 3, 4, 5, 6]
    source_assignment_chunks = {
        "galloping": [0, 0, 0],
        "isochronous": [0, 1, 0]
        }
    experiments = {}
    for n_rep in n_reps:
        for df in [4, 8]:
            for dt in [125]:
                for sk, source_assignment_chunk in source_assignment_chunks.items():
                    H = aba_hypothesis(
                        source_assignment_chunk, df, dt, n_rep, config, dB=55, A_Hz=500
                        )
                    observation_name = f"df{df:d}_dt{dt:d}_rep{n_rep:d}"
                    experiments.update(hutil.format(H, observation_name, sk))

    return experiments

######################
# Captor demos
######################


def captor_hypothesis(source_assignment, freq_captor, level_captor, config):

    tone_a = (freq_to_ERB(2200), 60)
    tone_b = (freq_to_ERB(2400), 60)
    distractor = (freq_to_ERB(1460), 65)

    if freq_captor is not None:
        captor = (freq_to_ERB(freq_captor), level_captor)
    
    tone_duration = 0.070
    if freq_captor is not None:
        fa_vals = [
            captor, captor, captor, distractor,
            tone_b, tone_a, distractor, captor, captor
            ]
        gaps = [
            0.009+0.064, 0.009+0.064, 0.009+0.064, 0.009+0.064, 
            0.009, 0.009, 0.009, 0.009+0.064, 0.009+0.064
            ]
        initial_timing = 0.1 + 0.009
    else:
        fa_vals = [distractor, tone_b, tone_a, distractor]
        gaps = [0.009, 0.009, 0.009, 0.009]
        initial_timing = (0.009+tone_duration+0.064)*3+0.1+0.009

    events = []
    for event_idx, fa_val in enumerate(fa_vals):
        event = {}
        event["gap"] = gaps[event_idx] if event_idx > 0 else initial_timing
        event["duration"] = tone_duration
        event["f0"] = fa_val[0]
        event["amplitude"] = fa_val[1]
        events.append(event)

    return create_scene_hypothesis(events, source_assignment, config)


def captor_design(config):

    experiments = {}
    observation_name = "compdown_fnone"
    experiments.update(hutil.format(
        captor_hypothesis([0, 1, 1, 0], None, None, config),
        observation_name,
        "C"
    ))
    experiments.update(hutil.format(
        captor_hypothesis([0, 0, 0, 0], None, None, config),
        observation_name,
        "D"
    ))

    source_assignments = {
        "C": [0, 0, 0, 0, 1, 1, 0, 0, 0],
        "D": [0, 0, 0, 1, 1, 1, 1, 0, 0],
        "3S": [0, 0, 0, 1, 2, 2, 1, 0, 0]
        }
    for freq_captor, level_captor in [(590, 63), (1030, 60), (1460, 65)]:
        for sk, source_assignment in source_assignments.items():
            H = captor_hypothesis(
                source_assignment, freq_captor, level_captor, config
                )
            observation_name = f"compdown_f{freq_captor:d}"
            experiments.update(hutil.format(H, observation_name, sk))       

    return experiments

######################
# Competition demos
######################


def abxy_hypothesis(source_assignment_chunk, abxy, levels, config, n_reps):

    source_assignment = np.tile(source_assignment_chunk, n_reps)
    tones = list(zip(abxy, levels))
    tone_duration = 0.1

    fa_vals = []
    for r in range(n_reps):
        fa_vals.extend(tones)

    events = []
    for event_idx, fa_val in enumerate(fa_vals):
        event = {}
        event["f0"] = freq_to_ERB(fa_val[0])
        event["amplitude"] = fa_val[1]
        event["gap"] = 0.050 if event_idx == 0 else 0.010
        event["duration"] = tone_duration
        events.append(event)

    return create_scene_hypothesis(events, source_assignment, config)


def competition_design(config, overwrite):

    tone_idxs = list(range(4))
    source_assignment_chunks = {}
    for n, p in enumerate(partition(tone_idxs), 1):
        org = []
        for i in range(4):
            where_i = np.where([(i in _p) for _p in p])[0][0]
            org.append(where_i)
        fn = "".join([str(o) for o in org])
        if fn in ["0123", "0111", "1011", "0001", "1101"]:
            continue
        source_assignment_chunks[fn] = org
    # See corresponding gen file
    levels = {'2800': 70, '2642': 70, '1556': 70, '1468': 70, '600': 70, '566': 70, '333': 77, '314': 77}
    frequencies = [
        [2800, 1556, 600, 333],
        [600, 333, 2800, 1556],
        [2800, 2642, 1556, 1468],
        [333, 314, 600, 566],
        [2800, 1556, 2642, 1468],
        [600, 333, 566, 314],
        [2800, 600, 1468, 314]
        ]
    conditions = ['isolate', 'isolate', 'isolate', 'isolate', 'absorb',' absorb', 'absorb']
    observation_names = []
    for i in range(len(conditions)):
        observation_names.append(f"{conditions[i]}_A{frequencies[i][0]}_B{frequencies[i][1]}_D1-{frequencies[i][2]}_D2-{frequencies[i][3]}")

    experiments = {}
    for observation_idx, observation_name in enumerate(observation_names):
        z = observation_name.split("_")
        A = int(z[1][1:])
        B = int(z[2][1:])
        X = int(z[3][3:])
        Y = int(z[4][3:])
        for sk, source_assignment_chunk in source_assignment_chunks.items():
            abxy_set = [A, B, X, Y]
            H = abxy_hypothesis(
                    source_assignment_chunk,
                    abxy_set,
                    [levels[str(f)] for f in abxy_set],
                    config,
                    n_reps=overwrite["n_repetitions"]
                )
            experiments.update(hutil.format(H, observation_name, sk))

    return experiments


# Helpers


def absolute_to_relative(events):
    offset = 0
    for event_idx in range(len(events)):
        events[event_idx]["gap"] = events[event_idx]["onset"] - offset
        events[event_idx]["duration"] = events[event_idx]["offset"] - events[event_idx]["onset"]        
        offset = events[event_idx]["offset"]
    return events


def relative_to_absolute(events):
    offset = 0
    for event_idx in range(len(events)):
        events[event_idx]["onset"] = events[event_idx]["gap"]+offset
        events[event_idx]["offset"] = events[event_idx]["onset"] + events[event_idx]["duration"]
        offset = events[event_idx]["offset"]
    return events


def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller
