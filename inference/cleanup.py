#!/bin/bash
# The following line allows this file to be run directly as a slurm script (e.g. "sbatch cleanup.py")
# -----------------------
''''exec python -u -- ${home_dir}/inference/cleanup.py ${1+"$@"}; exit; # '''
# -----------------------

from itertools import combinations
from copy import deepcopy
import os

import torch
import torch.nn as nn
import numpy as np
import scipy.interpolate

import inference.io
import inference.optimize
from inference.serial import hypothesis_setup
from util.context import context

########
# Logic of order of cleanup rounds
########


def check_remove_skip(next_number, keep_idxs):
    if keep_idxs is not None:
        # This means that the original proposals were kept
        # and none of the remove proposals were chosen
        if all([(i in keep_idxs) for i in range(len(keep_idxs))]):
            # We probably don't need to remove:
            # skip the rest of the remove proposals
            return "cleanup_merge_sources"
        else:
            # Try another remove proposal
            return "cleanup_remove_" + str(next_number)
    else:
        return "cleanup_remove_" + str(next_number)


def get_next_round_type(prev_round_type, keep_idxs=None, use_cleanup_flag=True):
    if not use_cleanup_flag:
        return None
    if prev_round_type == "sequential_final":
        return "cleanup_remove_0"
    # 3 remove iterations
    if prev_round_type == "cleanup_remove_0":
        return check_remove_skip(1, keep_idxs)
    elif prev_round_type == "cleanup_remove_1":
        return check_remove_skip(2, keep_idxs)
    elif prev_round_type == "cleanup_remove_2":
        # At most 3 remove proposals!
        return "cleanup_merge_sources"
    # Then merge moves
    elif prev_round_type == "cleanup_merge_sources":
        return "cleanup_merge_events"
    elif prev_round_type == "cleanup_merge_events":
        return "cleanup_h2n"
    elif prev_round_type == "cleanup_h2n":
        return None
    else:
        raise NotImplementedError

#########
# Clean-up proposals
#########


def remove_sources(old_scene, config):
    """ From an optimized scene, remove one source to create a set of new scenes """
    new_scenes = []
    n_sources = old_scene.n_sources.item()
    if n_sources == 1:
        return new_scenes
    for source_idx in range(n_sources):
        new_scene = deepcopy(old_scene)
        new_scene.n_sources = old_scene.n_sources - 1
        new_scene.sources = nn.ModuleList([
            old_scene.sources[si] for si in range(n_sources) if si != source_idx
            ])
        new_scenes.append(new_scene)
    return new_scenes


def remove_events(expt_name=None, sound_group=None, sound_name=None, hypothesis_name=None, old_scene=None, config=None):
    """ From an optimized scene, remove one event to create a set of new scenes"""

    if old_scene is None:
        hypothesis_dir = os.path.join(
            os.environ["inference_dir"], expt_name, sound_group, sound_name, hypothesis_name, ""
            )
        old_scene, ckpt = inference.io.restore_hypothesis(hypothesis_dir)
        config = inference.io.get_config(hypothesis_dir)

    event_source_idxs = [[
        (source_idx, event_idx) for event_idx in range(old_scene.sources[source_idx].sequence.n_events)
    ] for source_idx in range(old_scene.n_sources)]
    event_source_idxs = [item for sublist in event_source_idxs for item in sublist]

    new_scenes = []
    for source_idx, event_idx in event_source_idxs:
        with torch.no_grad(), context(config):
            new_scene = deepcopy(old_scene)
            if new_scene.sources[source_idx].sequence.n_events == 1:
                # This will be covered by the source removal proposal
                continue
            new_scene.sources[source_idx].sequence.n_events = new_scene.sources[source_idx].sequence.n_events - 1
            new_scene.sources[source_idx].sequence.events = torch.nn.ModuleList([
                event for (eidx, event) in enumerate(new_scene.sources[source_idx].sequence.events) if eidx != event_idx
                ])
            for event_idx, event in enumerate(new_scene.sources[source_idx].sequence.events):
                event.event_idx = event_idx
            new_scenes.append(new_scene)

    return new_scenes


def harmonic_to_noise(expt_name=None, sound_group=None, sound_name=None, hypothesis_name=None, old_scene=None, config=None):
    """ From an optimized scene, change the low-frequency harmonics to noises to create a set of new scenes """
    if old_scene is None:
        hypothesis_dir = os.path.join(
            os.environ["inference_dir"], expt_name, sound_group, sound_name, hypothesis_name, ""
            )
        old_scene, ckpt = inference.io.restore_hypothesis(hypothesis_dir)
        config = inference.io.get_config(hypothesis_dir)
    new_noisy_scenes = []
    n_sources = old_scene.n_sources.item()
    for source_idx in range(n_sources):
        if "harmonic" in old_scene.sources[source_idx].source_type:
            # Take a sample
            new_scene = deepcopy(old_scene)
            with torch.no_grad(), context(config, batch_size=10):
                scores = old_scene.pq_forward()
                # Get the source wave corresponding to the harmonic source
                sample_idx = scores.argmax().item()
                # Only use the low frequency harmonics for this proposal
                if old_scene.sources[source_idx].gps["f0"].mean_module.mu[sample_idx].item() > 7:
                    continue
                source_wave_to_match = old_scene.sources[source_idx].source_wave[sample_idx, :].detach().cpu().numpy()
                # Get spectrum corresponding to the source
                am_and_filter = old_scene.sources[source_idx].renderer.AM_and_filter
                grid_mask = am_and_filter.shift_E(
                    torch.ones(
                        am_and_filter.extended_spectrum.shape
                        ), am_and_filter.f0_for_shift
                    )[sample_idx, :, :]
                L = am_and_filter.shift_E(
                    am_and_filter.tf_grid.permute(0, 2, 1),
                    am_and_filter.f0_for_shift,
                    V=-20
                    )[sample_idx, :, :]
                below_f0 = (grid_mask.sum(1) <= grid_mask.shape[1]-1)
                shifted_spectrum = (~below_f0) * (L.sum(1)/(1e-10+grid_mask.sum(1))) + below_f0*-20
                shifted_spectrum[~below_f0] = shifted_spectrum[~below_f0] - shifted_spectrum[~below_f0].mean()
            # Create a noise source based on the harmonic one
            new_scene.n_sources = torch.ones(1).long()
            new_scene.sources = nn.ModuleList([ new_scene.sources[si] for si in range(n_sources) if si == source_idx ])
            new_scene.sources[0].source_type = "noise"
            new_scene.sources[0].gps.pop("f0")
            # Set spectrum to be shifted.
            with torch.no_grad():
                inducing_points = old_scene.sources[source_idx].gps["spectrum"].feature.variational_strategy.inducing_points.detach().cpu().numpy().squeeze()
                current_points = old_scene.sources[source_idx].gps["spectrum"].feature.gp_x[0,:,0].detach().cpu().numpy()
                new_variational_points = np.interp(inducing_points, current_points, shifted_spectrum)
                new_scene.sources[0].gps["spectrum"].feature.variational_strategy._variational_distribution._variational_mean = torch.nn.Parameter(torch.Tensor(new_variational_points)[None, :])
                new_scene.sources[0].gps["spectrum"].feature.variational_strategy._variational_distribution._variational_stddev.fill_(0.1)
            with context(config):
                new_scene.update_observation(torch.Tensor(source_wave_to_match[None, :]), config["likelihood"])
            # Optimize the noise source to match the harmonic source
            new_scene, optimizers, schedulers, metrics, hypothesis_config, savepath = hypothesis_setup(
                sound_name, sound_group, expt_name,
                hypothesis_name=f"{hypothesis_name}/cleanup_h2n_{source_idx:02d}",
                experiment_config=config, scene_init=new_scene, device=inference.io.get_device(cuda_idx=0)
                )
            with context(config, batch_size=config["optimization"]["batch_size"]):
                elbo = inference.optimize.basic_loop_from_scene(new_scene, optimizers, schedulers, metrics, hypothesis_config, savepath=savepath, earlystop=True, round_type="cleanup_h2n")
            # Put the optimized source into a new scene
            if not np.isnan(elbo):
                with torch.no_grad(), context(config):
                    new_full_scene = deepcopy(old_scene)
                    new_full_scene.sources.append(new_scene.sources[0])  # append new source to the end of this Modulelist
                    new_full_scene.sources = torch.nn.ModuleList([source for (sidx, source) in enumerate(new_full_scene.sources) if sidx != source_idx])
                    new_noisy_scenes.append(new_full_scene)

    return new_noisy_scenes


def get_onset(event, sample_idx):
    return event.onset.timepoint[sample_idx].item()


def get_offset(event, sample_idx):
    return event.offset.timepoint[sample_idx].item()


def merge_sources(expt_name=None, sound_group=None, sound_name=None, hypothesis_name=None, old_scene=None, config=None):
    """ Put all the events of one source into another source (only the same source type) """
    from model.sequence import Sequence
    from model.feature import Feature

    if old_scene is None:
        hypothesis_dir = os.path.join(os.environ["inference_dir"], expt_name, sound_group, sound_name, hypothesis_name, "")
        old_scene, ckpt = inference.io.restore_hypothesis(hypothesis_dir)
        config = inference.io.get_config(hypothesis_dir)

    with torch.no_grad(), context(config, batch_size=10):
        scores = old_scene.pq_forward()
        sample_idx = torch.argmax(scores)

    new_scenes = []
    for pair in combinations(range(old_scene.n_sources.item()), 2):
        # Apply criteria for merging
        # 1) Need to be the same source type
        if old_scene.sources[pair[0]].source_type != old_scene.sources[pair[1]].source_type:
            continue

        # 2) If they're mostly simultaneous, then there's no point in merging
        events_i = old_scene.sources[pair[0]].sequence.events
        events_j = old_scene.sources[pair[1]].sequence.events

        duration_i = sum([
            get_offset(event, sample_idx)-get_onset(event, sample_idx) for event in events_i
            ])
        duration_j = sum([
            get_offset(event, sample_idx)-get_onset(event, sample_idx) for event in events_j
            ])

        total_overlapping_duration = 0
        for event_i in events_i:
            for event_j in events_j:
                # Check if there is complete overlap?
                # Get overlapping_duration
                a = max([get_onset(event_i,sample_idx), get_onset(event_j,sample_idx)])
                b = min([get_offset(event_i,sample_idx), get_offset(event_j,sample_idx)])
                if b - a > 0:
                    total_overlapping_duration += (b-a)

        if total_overlapping_duration/max(duration_i, duration_j) > 0.9:
            continue

        # Construct the merge
        new_scene = deepcopy(old_scene)
        new_hypothesis = {}

        all_onsets = []
        all_onsets.extend([(
                event_idx, 0, get_onset(events_i[event_idx], sample_idx),
                get_offset(events_i[event_idx],sample_idx)
            ) for event_idx in range(len(events_i))])
        all_onsets.extend([(
                event_idx, 1, get_onset(events_j[event_idx], sample_idx),
                get_offset(events_j[event_idx],sample_idx)
            ) for event_idx in range(len(events_j))])

        # Normal combination for events that is used during inference
        sorted_events = sorted(all_onsets, key=lambda t: t[2])
        new_hypothesis_events = []
        MINIMUM_DURATION = config["renderer"]["steps"]["t"]
        sources_of_included_events = []
        for event_idx, event_info in enumerate(sorted_events):
            proposed_duration = event_info[3] - event_info[2]
            if proposed_duration < MINIMUM_DURATION:
                continue             
            if len(new_hypothesis_events) > 0:
                onset_before_last_offset = event_info[2] < new_hypothesis_events[-1]["offset"]
                if onset_before_last_offset:
                    # Check if the last tone can handle being cut down
                    potential_new_offset_last = event_info[2] - 0.001
                    potential_new_duration_last = potential_new_offset_last - new_hypothesis_events[-1]["onset"]
                    if potential_new_duration_last >= MINIMUM_DURATION:
                        new_hypothesis_events[-1]["offset"] = potential_new_offset_last
                        event_to_add = {"onset": event_info[2], "offset": event_info[3]}
                    else:
                        # Check if the new tone can handle being cut down
                        # while keeping the last event at the minimum duration
                        potential_new_offset_last = new_hypothesis_events[-1]["onset"] + MINIMUM_DURATION
                        potential_new_onset_current = potential_new_offset_last + 0.001
                        potential_new_duration_current = event_info[3] - potential_new_onset_current
                        # Earlier event will take precedent if we can't cut it
                        # down, we'll throw the current event out
                        if potential_new_duration_current >= MINIMUM_DURATION:
                            new_hypothesis_events[-1]["offset"] = potential_new_offset_last
                            event_to_add = {"onset": potential_new_onset_current, "offset": event_info[3]}
                        else:
                            event_to_add = None
                else:
                    event_to_add = {"onset": event_info[2], "offset": event_info[3]}
            else:
                event_to_add = {"onset": event_info[2], "offset": event_info[3]}
            if event_to_add is not None:
                new_hypothesis_events.append({"onset": event_info[2], "offset": event_info[3]})
                sources_of_included_events.append(event_info[1])

        new_hypothesis["events"] = new_hypothesis_events

        source_type = old_scene.sources[pair[0]].source_type
        new_hypothesis["source"] = source_type

        new_hypothesis["features"] = {}; gp_types = old_scene.sources[pair[0]].gps.keys()
        source_to_use = pair[0] if duration_i > duration_j else pair[1]
        source_to_delete = pair[1] if duration_i > duration_j else pair[0]
        for gp_type in gp_types:
            new_hypothesis["features"][gp_type]={}
            if gp_type == "spectrum":
                if "sigma_residual" in config["hyperpriors"][source_type]["spectrum"]["kernel"]["parametrization"]:
                    raise Exception("We haven't implemented the multispectral version of the merge proposal.")                
                else:
                    # Use spectrum from the sound that has a longer duration
                    new_hypothesis["features"]["spectrum"]["x"] = new_scene.sources[source_to_use].gps["spectrum"].feature.gp_x[0,:,0].detach().cpu().numpy()
                    new_hypothesis["features"]["spectrum"]["y"] = new_scene.sources[source_to_use].gps["spectrum"].feature.y[sample_idx,:].detach().cpu().numpy()
            else:
                # Temporal
                gps_to_pull_from = []
                for p in pair:
                    gp = new_scene.sources[p].gps[gp_type]
                    x = gp.feature.gp_x[0, :, 0].detach().cpu().numpy()
                    y = gp.feature.y[sample_idx, :].detach().cpu().numpy()
                    gps_to_pull_from.append({"x": x, "y": y})

                # If there's a streak of events from the same source, use that whole chunk
                chunks = []
                for event_idx, event in enumerate(new_hypothesis_events):
                    which_source = sources_of_included_events[event_idx]
                    if event_idx == 0:
                        chunks.append([0, event["offset"], which_source])
                    elif which_source == sorted_events[event_idx - 1][1]:
                        # Last event and this event are from the same source
                        chunks[-1][1] = event["offset"]
                    else:
                        # Last event and this event are from different sources
                        chunks[-1][1] = (chunks[-1][1] + event["onset"])/2
                        chunks.append([chunks[-1][1] + 0.001, event["offset"], which_source])
                chunks[-1][1] = old_scene.scene_duration

                gp_hypothesis = {"x": [], "y": []}
                for onset, offset, which_source  in chunks:
                    gp_to_pull_from = gps_to_pull_from[which_source]
                    x_in_chunk = (onset <= gp_to_pull_from["x"]) * (gp_to_pull_from["x"] <= offset)
                    gp_hypothesis["x"].append(gp_to_pull_from["x"][x_in_chunk])
                    gp_hypothesis["y"].append(gp_to_pull_from["y"][x_in_chunk])

                new_hypothesis["features"][gp_type] = {"x": np.concatenate(gp_hypothesis["x"]), "y": np.concatenate(gp_hypothesis["y"])}

        # Delete pair[1] and alter pair[0] in new_scene
        with torch.no_grad(), context(config, scene=new_scene, batch_size=10):
            source_type = new_scene.sources[source_to_use].source_type
            new_scene.sources[source_to_use].sequence = Sequence.hypothesize(new_hypothesis, config["hyperpriors"])
            source_type = new_scene.sources[source_to_use].source_type
            for gp_type in config["hyperpriors"][source_type].keys():
                new_scene.sources[source_to_use].gps[gp_type].feature = Feature.hypothesize(new_hypothesis, config["hyperpriors"][source_type][gp_type], gp_type)            
            new_scene.sources = nn.ModuleList([source for source_idx, source in enumerate(new_scene.sources) if source_idx != source_to_delete])
            new_scene.n_sources = new_scene.n_sources - 1
            new_scenes.append(deepcopy(new_scene))

    return new_scenes


def merge_events(expt_name=None, sound_group=None, sound_name=None, hypothesis_name=None, old_scene=None, config=None):
    """ for every event, merge it with the subsequent event
        if it's at the end, extend it to the end
    """
    from model.sequence import Sequence
    from model.feature import Feature

    if old_scene is None:
        hypothesis_dir = os.path.join(os.environ["inference_dir"], expt_name, sound_group, sound_name, hypothesis_name, "")
        old_scene, ckpt = inference.io.restore_hypothesis(hypothesis_dir)
        config = inference.io.get_config(hypothesis_dir)
    minimum_merge_duration = config["heuristics"]["cleanup"]["minimum_merge_duration"]

    with torch.no_grad(), context(config, batch_size=10):
        scores = old_scene.pq_forward()
        sample_idx = torch.argmax(scores)

    new_scenes = []
    for source_idx, source in enumerate(old_scene.sources):
        for event_idx in range(source.sequence.n_events):
            # Find the timings corresponding to the merges we want to make
            if event_idx == source.sequence.n_events - 1:
                current_onset = get_onset(source.sequence.events[event_idx], sample_idx)
                if (old_scene.scene_duration - current_onset) > minimum_merge_duration:
                    new_timing = (current_onset, old_scene.scene_duration)
                    old_silence = (get_offset(source.sequence.events[event_idx], sample_idx), old_scene.scene_duration)
                else:
                    continue
            else:
                # If there's a large enough gap, try merging
                gap_between_current_and_next = get_onset(source.sequence.events[event_idx+1], sample_idx) - get_offset(source.sequence.events[event_idx],sample_idx)
                if gap_between_current_and_next > minimum_merge_duration:
                    new_timing = (get_onset(source.sequence.events[event_idx],sample_idx), get_offset(source.sequence.events[event_idx+1],sample_idx))
                    old_silence = (get_offset(source.sequence.events[event_idx],sample_idx), get_onset(source.sequence.events[event_idx+1],sample_idx))
                else:
                    continue

            # Construct the merge
            new_hypothesis = {}
            new_hypothesis["events"] = []
            for _event_idx, _event in enumerate(source.sequence.events):
                if _event_idx == event_idx:
                    new_hypothesis["events"].append({"onset": new_timing[0], "offset": new_timing[1]})
                elif _event_idx == event_idx + 1:
                    continue
                else:
                    new_event_to_append = {"onset": get_onset(_event, sample_idx), "offset": get_offset(_event, sample_idx)}
                    if new_event_to_append["onset"] > old_scene.scene_duration:
                        continue
                    else:
                        new_hypothesis["events"].append(new_event_to_append)

            new_hypothesis["features"] = {}
            gp_types = source.gps.keys()
            source_type = source.source_type
            for gp_type in gp_types:
                new_hypothesis["features"][gp_type] = {}
                if gp_type == "spectrum":
                    if "sigma_residual" in config["hyperpriors"][source_type]["spectrum"]["kernel"]["parametrization"]:
                        raise Exception("We haven't implemented the multispectral version of the merge proposal.")                
                    else:
                        # Use spectrum from the sound that has a longer duration
                        new_hypothesis["features"]["spectrum"]["x"] = source.gps["spectrum"].feature.gp_x[0,:,0].detach().cpu().numpy()
                        new_hypothesis["features"]["spectrum"]["y"] = source.gps["spectrum"].feature.y[sample_idx,:].detach().cpu().numpy()
                else:
                    # Temporal
                    x = source.gps[gp_type].feature.gp_x[0,:,0].detach().cpu().numpy() 
                    y = source.gps[gp_type].feature.y[sample_idx,:].detach().cpu().numpy() 

                    # Define the gp values in what used to be silence
                    x_in_silence = x[(old_silence[0] < x)*(x < old_silence[1])]

                    # Get only the "visible" points and then extrapolate into the old silence
                    old_active = {"x": [], "y": []}
                    for _event_idx, _event in enumerate(source.sequence.events):
                        x_in_event = (get_onset(_event, sample_idx) <= x) * (x <= get_offset(_event, sample_idx))
                        old_active["x"].extend(x[x_in_event])
                        old_active["y"].extend(y[x_in_event])

                    # Get the points in the silence through "nearest" interpolation
                    interpol8r = scipy.interpolate.interp1d(
                        np.array(old_active["x"]), np.array(old_active["y"]),
                        kind="nearest", fill_value="extrapolate"
                        )
                    new_points_in_silence = interpol8r(x_in_silence)

                    # Put the old and new points together in the new hypothesis
                    new_x = np.concatenate((np.array(old_active["x"]), x_in_silence))
                    sort_x = np.argsort(new_x)
                    new_x = new_x[sort_x]
                    new_y = np.concatenate((np.array(old_active["y"]), new_points_in_silence))
                    new_y = new_y[sort_x]
                    new_hypothesis["features"][gp_type] = {"x": new_x, "y": new_y}

            new_scene = deepcopy(old_scene)
            with torch.no_grad(), context(config, scene=new_scene, batch_size=10):
                new_scene.sources[source_idx].sequence = Sequence.hypothesize(new_hypothesis, config["hyperpriors"])
                source_type = new_scene.sources[source_idx].source_type
                for gp_type in config["hyperpriors"][source_type].keys():
                    new_scene.sources[source_idx].gps[gp_type].feature = Feature.hypothesize(new_hypothesis, config["hyperpriors"][source_type][gp_type], gp_type)
                new_scenes.append(deepcopy(new_scene))

    return new_scenes
