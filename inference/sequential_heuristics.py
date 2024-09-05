import numpy as np

# Implement heuristics for sequential inference
# See Appendix B, section B.2.2, Source construction


def permissible(scene_hypotheses, config, round_idx):
    """ Return scenes which are permitted according to the heuristics """

    # Max number of sources
    max_sources = config["heuristics"]["sequential"]["max_sources"]
    scene_hypotheses = [scene for scene in scene_hypotheses if len(scene.sources) <= max_sources]

    # Max number of new sources
    max_new_sources = config["heuristics"]["sequential"]["max_new_sources_later"] if round_idx > 0 else config["heuristics"]["sequential"]["max_new_sources_0"]
    scene_hypotheses = [scene for scene in scene_hypotheses if len(scene.meta.new_sources) <= max_new_sources]

    # Max number of new events
    max_new_events = config["heuristics"]["sequential"]["max_new_events"]
    scene_hypotheses = [scene for scene in scene_hypotheses if len(scene.meta.new_events) <= max_new_events]

    # Min event score
    min_event_score = config["heuristics"]["sequential"].get("min_event_score", 0)
    scene_hypotheses = [
            scene for scene in scene_hypotheses if all(
                event.meta.background_proposal or event.meta.seg_score > min_event_score
                for source in scene.sources for event in source.sequence.events
            )
        ]

    # Max event rank
    max_event_rank = config["heuristics"]["sequential"].get("max_event_rank", np.Inf)
    scene_hypotheses = [
        scene for scene in scene_hypotheses if all(
                event.meta.background_proposal or event.meta.proposal_rank <= max_event_rank
                for source in scene.sources for event in source.sequence.events
            )
        ]

    # After round X, never include “add on its own” proposals
    if round_idx >= config["heuristics"]["sequential"]["only_new_before"]:
        scene_hypotheses = [scene for scene in scene_hypotheses if len(scene.meta.history) > 1]

    return scene_hypotheses


def prioritize(scene_hypotheses, config, round_idx):
    """ Sort scenes depending on heuristics """
    scene_hypotheses = permissible(scene_hypotheses, config, round_idx)

    max_scene_hypotheses = config["heuristics"]["sequential"]["max_scenes_per_iter"]

    def event_score(scene_hypothesis):
        new_events, _ = get_max_iou_with_new(scene_hypothesis, round_idx)
        new_event_scores = [event.meta.seg_score for event in new_events]
        min_new_event_score = np.min(new_event_scores) if len(new_event_scores) > 0 else np.inf
        avg_new_event_score = np.mean(new_event_scores) if len(new_event_scores) > 0 else np.inf
        return (min_new_event_score, avg_new_event_score)

    def get_rank_dict(scene_hypotheses, f):
        score_dict = {scene_hypothesis: f(scene_hypothesis) for scene_hypothesis in scene_hypotheses}
        scores = sorted(list(score_dict.values()), reverse=True)
        return {k: scores.index(v) for k, v in score_dict.items()}

    def iou_threshold(scene_hypothesis):
        _, max_iou = get_max_iou_with_new(scene_hypothesis, round_idx)
        return max_iou < 0.1

    rank1 = get_rank_dict(scene_hypotheses, event_score)
    rank2 = get_rank_dict(scene_hypotheses, lambda x: (iou_threshold(x), event_score(x)))

    scene_hypotheses = sorted(scene_hypotheses, key=lambda x: min(rank1[x], rank2[x]))[:max_scene_hypotheses]

    return scene_hypotheses


def get_max_iou_with_new(scene, current_round_idx):
    """ Measure overlap of new events with existing evnets """
    # Get events added before this round (old) and events added this round (new)
    old = []
    new = []
    for source in scene.sources:
        for event in source.sequence.events:
            if event.meta.round_idx == current_round_idx:
                new.append(event)
            else:
                old.append(event)
    # Find the iou of the new events with everything else
    all_new_ious = []
    for i in range(len(new)):
        for j in range(i+1,len(new)):
            includes_background = new[i].meta.background_proposal or new[j].meta.background_proposal
            iou = new[i].meta.ious[new[j].meta.proposal_rank] if not includes_background else 0.0
            all_new_ious.append(iou)
    for i in range(len(new)):
        for j in range(len(old)):
            includes_background = new[i].meta.background_proposal or old[j].meta.background_proposal
            iou = new[i].meta.ious[old[j].meta.proposal_rank] if not includes_background else 0.0
            all_new_ious.append(iou)
    return new, np.max(all_new_ious) if len(all_new_ious) > 0 else 0


def check_event_fit(self, event_proposal, config):
    """ Check if the temporal overlap between events is permissible """ 
    event = self.events[-1]
    if event.offset._mu.item() < event_proposal["onset"]:
        """simple case where the proposed event is after the last event of the sequence"""
        b = True
    elif (event.onset._mu.item() + event.offset.low_limit.item()) < event_proposal["onset"] <= event.offset._mu.item() < event_proposal["offset"]:
        """situation :
        --a--  ---b---
                    --c--
        result ==> cut off b early (b remains above the acceptable low_limit):
        --a--  ---b-
                    --c--
        """
        intersection = min(
                event_proposal["offset"], event.offset._mu.item()
            ) - max(
                event_proposal["onset"], event.onset._mu.item()
            )
        # Max checks for are these two almost identical
        # Min would check if one contains the other
        proportion = intersection / max(
                (event_proposal["offset"] - event_proposal["onset"]),
                (event.offset._mu.item() - event.onset._mu.item())
            )
        if proportion > config["event_fit"]["max_proportion"]:
            # These two occupy nearly identical time intervals
            # one of them would be cut down to basically nothing.
            b = False
        else:
            b = True
    elif event.onset._mu.item() <= event_proposal["onset"] < event_proposal["offset"] <= event.offset._mu.item():
        """situation:           .... or 
        --a--  ----b----        ----b----
                --c--          ----c--
        """
        b = False
    else:
        b = False
    return b
