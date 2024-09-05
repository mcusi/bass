import json
import os

import torch

import inference.io
import inference.sequential_heuristics
import inference.proposals
import inference.optimize
import inference.cleanup
from util.context import context
from model.scene import Scene

# Creating new scene hypotheses from previous scenes and event proposals
# see Appendix B > Sequential inference > Source construction

###################
# Helper functions
###################


def get_hypothesis_name(round_idx, hypothesis_number, round_type="sequential"):
    """ Standardize naming for hypotheses """
    if round_idx == "*" and hypothesis_number == "*":
        return f"round*-*"
    elif round_idx == "*":
        return f"round*-{hypothesis_number:03d}"
    elif hypothesis_number == "*":
        return f"round{round_idx:03d}-*"
    else:
        return f"round{round_idx:03d}-{hypothesis_number:03d}"


def analyze_previous_round(hypotheses_dir, round_idx):
    """ Analyze complete round=round_idx of iterative inference
        to pass on the top N hypotheses to round=round_idx+1
    """

    # Save results
    results_fn = hypotheses_dir + "results.json"
    try:
        with open(results_fn, 'r') as outfile:
            saved_results = json.load(outfile)
        keep_idxs = saved_results["keep"][round_idx-1]
        reject_idxs = saved_results["reject"][round_idx-1]
    except:
        raise Exception(f"results.json is inconsistent with current round = {round_idx}!")

    print(f"On round {round_idx-1}: keep=", keep_idxs, ", reject=", reject_idxs, flush=True)

    return keep_idxs

###################
# Creating new scene hypotheses
###################


def create_new_scene_hypotheses(new_event_proposals, keep_idxs, round_idx, experiment_config, observation, audio_sr, hypotheses_dir):
    """ Create scene hypotheses from event
        proposals and previous pairs if any
    """

    # Load in hypotheses from previous rounds
    scene_hypotheses = []
    if round_idx > 0:
        for keep_idx in keep_idxs:
            previous_savepath = os.path.join(hypotheses_dir, get_hypothesis_name(round_idx-1, keep_idx), "")
            scene_init = torch.load(os.path.join(previous_savepath, "scene_structure.tar"), map_location="cpu")
            scene_opt, _ = inference.io.restore_hypothesis(previous_savepath)
            scene, _ = inference.proposals.reset_last_sequence_offset(scene_opt, scene_init, scene_opt.scene_duration)
            scene_hypotheses.append(scene)

    # Add the proposals into the hypotheses and update them to use the trimmed observation
    with context(experiment_config, batch_size=experiment_config["optimization"]["batch_size"]):
        # 1) Update old hypotheses to have new observation
        for scene_hypothesis in scene_hypotheses:
            scene_hypothesis.update_observation(observation, experiment_config["likelihood"])
            scene_hypothesis.meta.next_round()

        # 2) Add new event proposals to old hypotheses
        for proposal_idx, new_event_proposal in new_event_proposals:
            print("Adding in a new element proposal: ", proposal_idx)  
            candidate_scene_hypotheses = [] 
            for prev_hypothesis in list(scene_hypotheses):
                candidate_scene_hypotheses.extend(inference.proposals.add_event_to_scene(prev_hypothesis, new_event_proposal, experiment_config, observation, audio_sr))
            # 3) Create hypothesis which only has a single new event proposal in it
            candidate_scene_hypotheses.append(Scene.hypothesize([new_event_proposal], experiment_config["hyperpriors"], experiment_config["likelihood"], observation, audio_sr))

            # Relies on `prioritize` being applicable only to a single batch of hypotheses, and valid for all batches
            scene_hypotheses.extend(inference.sequential_heuristics.permissible(candidate_scene_hypotheses, experiment_config, round_idx))
            scene_hypotheses = inference.sequential_heuristics.prioritize(scene_hypotheses, experiment_config, round_idx)

    return scene_hypotheses


def create_scene_hypotheses_cleanup(sound_name, sound_group, expt_name, experiment_config, round_type, round_idx):
    """ Creating scene hypotheses based on cleanup proposals, see Appendix B > Sequential inference > Cleanup proposals """

    hypotheses_dir = os.path.join(os.environ["inference_dir"], expt_name, sound_group, sound_name, "")
    # Load in hypotheses from previous rounds 
    keep_idxs = analyze_previous_round(hypotheses_dir, round_idx)
    scene_hypotheses = []; hypothesis_names = []; previous_savepaths = []
    for keep_idx in keep_idxs:
        hypothesis_name = get_hypothesis_name(round_idx-1, keep_idx)
        previous_savepath = hypotheses_dir + hypothesis_name + "/"
        scene, _ = inference.io.restore_hypothesis(previous_savepath)
        scene_hypotheses.append(scene)
        scene.meta.next_round(round_type)
        hypothesis_names.append(hypothesis_name)
        previous_savepaths.append(previous_savepath)

    if "cleanup_remove" in round_type:
        # Remove sources
        remove_list = []
        for scene in scene_hypotheses:
            remove_list.extend(inference.cleanup.remove_sources(old_scene=scene, config=experiment_config))
            remove_list.extend(inference.cleanup.remove_events(old_scene=scene, config=experiment_config))
        scene_hypotheses.extend(remove_list)

    elif round_type == "cleanup_merge_sources":
        # Merge sources of the same type
        merge_list = []
        for scene in scene_hypotheses:
            merge_list.extend(inference.cleanup.merge_sources(old_scene=scene, config=experiment_config))
        scene_hypotheses.extend(merge_list)

    elif round_type == "cleanup_merge_events":
        # Merge subsequent events within a source
        merge_list = []
        for scene in scene_hypotheses:
            merge_list.extend(inference.cleanup.merge_events(old_scene=scene, config=experiment_config))
        scene_hypotheses.extend(merge_list)

    elif round_type == "cleanup_h2n":
        # Turn low-frequency harmonics into noises
        h2n_list = []
        for scene_idx, scene in enumerate(scene_hypotheses):
            h2n_list.extend(inference.cleanup.harmonic_to_noise(expt_name=expt_name, sound_group=sound_group, sound_name=sound_name, hypothesis_name=hypothesis_names[scene_idx], old_scene=scene, config=experiment_config))
        scene_hypotheses.extend(h2n_list)

    else:
        raise NotImplementedError()

    return scene_hypotheses, previous_savepaths
