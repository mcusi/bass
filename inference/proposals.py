import os
from copy import deepcopy

import numpy as np 
import torch

import renderer.util
from model.scene import Scene
import inference.sequential_heuristics
from util.context import context

"""
Functions for integrating event proposals into Scene objects
Most event proposals are neural event proposals, i.e. output from the neural network

Assumed structure for event proposal mimics that for source hypotheses, but limited to one event:
{"source_type": str
 "events":[ ==> if you want to change this to "event":{"onset", "offset", "meta"}, please search ["events"][0] to change these!
    { "onset": float
      "offset": float
      "meta": {"rank": int, "ious": [list of floats]} ==> provides information on how this proposal relates to others.
    }
 ],
 "features":{
     str: {"x": array, "y": array}
 }
}

Since it mimics scene hypothesis, we can use it directly in Scene.hypothesize to begin a hypothesis.
"""

####################
# Handling event proposals from neural network
####################


def load_event_proposals_and_sound(demo_name, audio_folder, amortize, source_types_to_keep=None, minimum_duration=0.0):
    """ Load in network outputs, ensure they're in the right format"""

    # Get detectron outputs
    detectron_output_fn = os.path.join(
        os.environ["segmentation_dir"], amortize["segmentation"], audio_folder, demo_name + ".tar"
        )
    detectron_tar = torch.load(detectron_output_fn, map_location="cpu")
    iou_matrix = detectron_tar["outputs"]["iou_matrix"]
    net_proposals = detectron_tar["outputs"]["latents"]

    event_proposals = []
    for net_proposal in net_proposals:
        event_proposal = {}
        if source_types_to_keep is not None:
            # in which case keep ALL source types
            if net_proposal["source_type"] not in source_types_to_keep:
                # This lets us set the source type args in the config
                # in order to restrict inference to certain source types
                # if needed
                continue
        event_proposal["source_type"] = net_proposal["source_type"]
        rank = net_proposal["rank"]
        if net_proposal["offset"] - net_proposal["onset"] < minimum_duration:
            continue
        event_proposal["events"] = [{
            "onset": net_proposal["onset"],
            "offset": net_proposal["offset"],
            "meta": {
                "score": detectron_tar["outputs"]["scores"][rank].item(),
                "rank": rank,
                "ious": iou_matrix[rank, :].tolist()
                }
            }]
        event_proposal["features"] = net_proposal["gps"]
        if "spectrum" in event_proposal["features"].keys():
            event_proposal["features"]["spectrum"]["x_events"] = [event_proposal["features"]["spectrum"]["x"]]
            event_proposal["features"]["spectrum"]["y_events"] = [event_proposal["features"]["spectrum"]["y"]]
        event_proposals.append(event_proposal)
    return event_proposals, detectron_tar["scene_wave"][0, :], detectron_tar["audio_sr"]


def group_event_proposals(event_proposals, config):
    """Make subgroups of event proposals that are included at the same seq. inference step
    """
    # Sort network outputs by onset
    # Sorts are guaranteed to be stable
    sorted_event_proposals = sorted(event_proposals, key=lambda k: k["events"][0]["onset"])
    onset_buffer = config["optimization"]["onset_buffer"]
    max_concurrent = config["optimization"]["max_concurrent"]
    # List of lists - integers organized as [ [a_0, a_1, ..., a_n], [b_0, ...], ... ]
    # such that all the a-onsets are within onset_buffer of a_0
    grouped_idxs = []
    for i in range(len(sorted_event_proposals)):
        if any([i in gdi for gdi in grouped_idxs]):
            continue
        else:
            grouped_idxs.append([i])
            if i == len(sorted_event_proposals) - 1:
                break
            this_onset = sorted_event_proposals[i]["events"][0]["onset"]
        for j in range(i + 1, len(sorted_event_proposals)):
            if this_onset <= sorted_event_proposals[j]["events"][0]["onset"] <= this_onset + onset_buffer:
                grouped_idxs[-1].append(j)

    # If there are too many events happening currently, drop the lower ranked ones
    # ranked_idxs:
    # List of lists of ints - similar to grouped_idxs
    # but the maxlen of any list is determined by max_concurrent
    ranked_idxs = []
    # List of event proposals - flattened 'rank_idx' and got those event_proposals --> [a_0, a_1, ..., b_0, ...]
    event_proposal_sequence = []
    for round_idx, idx_group in enumerate(grouped_idxs):
        if len(idx_group) <= max_concurrent:
            # Just keep all the event proposals at this time window
            ranked_idxs.append(idx_group)
            for i in idx_group:
                sorted_event_proposals[i]["events"][0]["meta"]["round_idx"] = round_idx
            event_proposal_sequence.extend([sorted_event_proposals[i] for i in idx_group])
        else:
            # Sort by amortized ranking
            r = sorted(idx_group, key=lambda i: sorted_event_proposals[i]["events"][0]["meta"]["rank"])
            keep_idxs = sorted(r[:max_concurrent])  # Re-sort by onset
            ranked_idxs.append(keep_idxs)
            for max_concurrent_rank, i in enumerate(r[:max_concurrent]):
                sorted_event_proposals[i]["events"][0]["meta"]["concurrent_rank"] = max_concurrent_rank
                sorted_event_proposals[i]["events"][0]["meta"]["round_idx"] = round_idx
            event_proposal_sequence.extend([sorted_event_proposals[i] for i in keep_idxs])

    # Get the changing scene_duration over iterations of the sequential inference algorithm
    # Do not change the actual onset in the initialization,
    # this is just for the sake of combining elems & setting scene_duration
    # event_group_onset_sequence: [onset_a_0-ramp, onset_a_0-ramp, ..., onset_b_0-ramp, ...]
    event_group_onset_sequence = []
    ramp_duration = config["renderer"]["ramp_duration"]
    for round_idx, idx_group in enumerate(ranked_idxs):
        # Need to subtract because "onset" refers to the end of the ramp rise
        # If we don't do this, then the last existing element of the scene will
        # be forced to explain the ramp up of the next new element
        group_onset = sorted_event_proposals[idx_group[0]]["events"][0]["onset"] - ramp_duration
        event_group_onset_sequence.extend([group_onset  for _ in range(len(idx_group))])

    # Add the group onset to the event meta
    for event_idx, group_onset in enumerate(event_group_onset_sequence):
        event_proposal_sequence[event_idx]["events"][0]["meta"]["group_onset"] = group_onset

    return event_proposal_sequence


####################
# Add events to scenes/sources
####################


def add_event_to_source(old_scene, source_idx, event_proposal, hyperpriors, config):
    """Add an event to the end of an existing Source in a Scene 
    Requires torch.no_grad"""
    fits = old_scene.sources[source_idx].check_fit(event_proposal)
    if fits:
        new_scene = deepcopy(old_scene)
        new_scene.sources[source_idx].update(event_proposal, hyperpriors, config)
        return new_scene
    else:
        return None


def add_event_to_scene(old_scene, event_proposal, config, observation, audio_sr):
    """Creates all allowed combinations of event_proposal with old_scene
    Must occur within a `context` with upcoming scene_duration (bc of `set_active` in GP)"""
    new_scenes = []
    # Check if this proposal has too much overlap
    # with any of the existing events in the scene
    iou_pass = True if ("meta" not in event_proposal["events"][0].keys()) else old_scene.check_iou(event_proposal)
    if iou_pass:
        with torch.no_grad():
            # 1. Add event proposal to existing sources
            for source_idx in range(old_scene.n_sources):
                if hasattr(old_scene.sources[source_idx], "background"):
                    continue
                new_scene = add_event_to_source(old_scene, source_idx, event_proposal, config["hyperpriors"], config)
                if new_scene is not None:
                    new_scenes.append(new_scene)
            # 2. Create a new source with event proposal
            new_scene = deepcopy(old_scene)
            new_scene.update(event_proposal, config["hyperpriors"])
            new_scenes.append(new_scene)
            # After this, update renderers with new scene_duration as well
            for new_scene in new_scenes:
                new_scene.update_observation(observation, config["likelihood"])
    return new_scenes


def reset_last_sequence_offset(scene_opt, scene_init, scene_duration):
    """
        Resets last offset of sequence in case the truncated
        optimization did something unreasonable to it (eg because unobserved)
    """
    with torch.no_grad():
        for source_idx in range(scene_init.n_sources.item()):
            events_opt = scene_opt.sources[source_idx].sequence.events
            events_init = scene_init.sources[source_idx].sequence.events

            if events_init[-1].offset._mu.item() > scene_duration:
                # If offset was initialized to beyond the scene,
                # snap back to initialization
                scene_opt.sources[source_idx].sequence.events[-1].offset._mu = deepcopy(
                    scene_init.sources[source_idx].sequence.events[-1].offset._mu
                    )
                scene_opt.sources[source_idx].sequence.events[-1].offset._sigma = deepcopy(
                    scene_init.sources[source_idx].sequence.events[-1].offset._sigma
                    )
            elif events_opt[-1].offset._mu.item() > scene_duration:
                # If offset MOVED to beyond the scene, then snap to the scene boundary
                # ie., events_init[-1].offset._mu.item() < scene_duration
                device = scene_opt.sources[source_idx].sequence.events[-1].offset._mu.device
                scene_opt.sources[source_idx].sequence.events[-1].offset._mu.fill_(scene_duration)
                scene_opt.sources[source_idx].sequence.events[-1].offset._sigma = deepcopy(
                    scene_init.sources[source_idx].sequence.events[-1].offset._sigma
                    )

    return scene_opt


####################
# Background noise proposal
####################


def background_noise_proposal(scene_duration, audio_sr, config):
    """
        Define the latent variables for a background noise proposal,
        which is a constant, moderate level noise lasting the entire scene
    """

    fs = renderer.util.get_event_gp_freqs(
        audio_sr,
        config["hypothesis"]["delta_gp"],
        lo_lim_freq=config["renderer"]["lo_lim_freq"]
        )
    ts = np.arange(
        0.001,
        scene_duration + config["hypothesis"]["delta_gp"]["t"],
        config["hypothesis"]["delta_gp"]["t"]
        )

    background_proposal = {
        "source_type": "noise",
        "events":[{
            "onset": 0.0,
            "offset": scene_duration,
            "meta": {"score": -1,
                     "rank": -1,
                     "ious": None,
                     "background_proposal": True
                     }}],
        "features":{
            "spectrum": {
                "x": fs,
                "y": np.random.randn(*fs.shape)},
            "amplitude": {
                "x": ts,
                "y": np.full(ts.shape, 40.0)}}}

    background_proposal["features"]["spectrum"]["x_events"] = [background_proposal["features"]["spectrum"]["x"]]
    background_proposal["features"]["spectrum"]["y_events"] = [background_proposal["features"]["spectrum"]["y"]]

    return background_proposal


def add_background_proposal(scene_hypotheses, round_idx, experiment_config, observation, audio_sr, full_scene_duration):
    """ Add a background proposal to the zeroth round, potentially doubling the number of proposals (no prioritization applied) """
    with context(experiment_config, batch_size=experiment_config["optimization"]["batch_size"]):
        # Create background proposal
        background_proposal = background_noise_proposal(full_scene_duration, audio_sr, experiment_config)
        # Add background to all previous scenes
        candidate_scene_hypotheses = []
        for prev_hypothesis in list(scene_hypotheses):
            candidate_scene_hypotheses.extend(add_event_to_scene(prev_hypothesis, background_proposal, experiment_config, observation, audio_sr))
        # Permissible rules apply, but not prioritization.
        scene_hypotheses.extend(inference.sequential_heuristics.permissible(candidate_scene_hypotheses, experiment_config, round_idx))
        # Add one scene with only background
        scene_hypotheses.append(Scene.hypothesize([background_proposal], experiment_config["hyperpriors"], experiment_config["likelihood"], observation, audio_sr))
    return scene_hypotheses