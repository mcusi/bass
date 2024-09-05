import tqdm
import numpy as np
import traceback

import torch
import torch.optim

import inference.io
import inference.metrics
from util.context import context

##################
# Learn meta-source parameters, which define source priors.
# See Appendix A > Generative model: Source priors
##################


def metasource_variational_update(scenes, optimizers, schedulers, metrics, chunk_size=3):
    """ One iteration of sampling from Scenes and applying gradient updates to learn meta-source parameters """

    # Shuffle which scenes are optimized together
    idx = np.random.randn(*scenes.scene_assignments.shape).argsort(0)
    scenes.scene_assignments = scenes.scene_assignments[idx, np.arange(scenes.scene_assignments.shape[1])]
    if scenes.scene_assignments.shape[1] == 1:
        # only one device
        chunks = [
            scenes.scene_assignments[x:x+chunk_size, 0] for x in range(0, len(scenes.scene_assignments), chunk_size)
            ]
    else:
        chunks = [list(scenes.scene_assignments[j, :]) for j in range(scenes.scene_assignments.shape[0])]

    # Optimization split into chunks to make memory usage manageable
    for optimizer in optimizers:
        optimizer.zero_grad()
    for scene_idxs in chunks:
        # Run forward and backward
        scores = scenes.forward(scene_idxs)
        loss = inference.metrics.mean_score(scores)
        loss.backward()
        metrics.step_update(loss, scores)
    for optimizer in optimizers:
        optimizer.step()

    # Update scheduler only after all the scenes have been seen
    metrics.epoch_update(scenes, len(chunks))
    if schedulers is not None:
        schedule_cycle = context.optimization["schedule_cycle"]
        if (len(metrics.epoch_loss) % schedule_cycle == 0) and (len(metrics.epoch_loss) >= schedule_cycle):
            for scheduler in schedulers:
                scheduler.step(metrics.epoch_loss[-1])


def infer(scenes, optimizers, schedulers, metrics, config, savepath="./"):
    """ Perform inference of meta-source parameters using a Scenes object.
        Requires `with context(...):` at level above.
    """
    print("Starting inference from step ", len(metrics.epoch_loss))
    taq_iterator = tqdm.tqdm(
        range(
            len(metrics.epoch_loss),
            config["optimization"]["total_steps"] + 1
        )
    )

    # Run everything forward once & set likelihood masks if necessary
    with torch.no_grad():
        for scene in scenes.scenes:
            scene.pq_forward()
            if config["renderer"]["tf"].get("likelihood_mask", False):
                scene.set_likelihood_mask()

    for step_idx in taq_iterator:
        metasource_variational_update(scenes, optimizers, schedulers, metrics)

        # Log various metrics and plot scenes
        if step_idx % config["report"]["print"] == 0:
            taq_iterator.set_description("Epoch {:05d} | Epoch loss {:.1f}".format(step_idx, metrics.epoch_loss[-1] ))

        if (step_idx % config["report"]["ckpt"] == 0) or (step_idx == config["optimization"]["total_steps"]):
            inference.io.save_state_hp(savepath, scenes, step_idx, metrics, optimizers, schedulers)

        if step_idx % config["report"]["plot"] == 0:
            metrics.plot(savepath)

    print("Inference complete!")
    return metrics.epoch_loss[-1]

#########################################
# To obtain source priors for F0, we
# conditioned on ground truth pitch tracks
# during inference. This means, instead of
# rendering sounds during inference, we
# conditioned on the event-level variables
# of f0 trajectory to infer the source priors.
# The next two functions carry out this inference.
# See Appendix A > Generative model: Source priors > Gaussian process source priors
#########################################


def infer_from_events(gp_type, scenes, optimizers, schedulers, metrics, config, savepath="./"):
    """ Perform inference of meta-source parameters using a Scenes
        object conditioned on event-level variables, without rendering
        Requires `with context(...):` at level above.
    """

    with torch.no_grad():
        scenes.forward(range(len(scenes.device_assignments)))
    print("Starting inference from step ", len(metrics.loss))
    taq_iterator = tqdm.tqdm(range(
        len(metrics.loss), config["optimization"]["total_steps"] + 1
        ))

    for step_idx in taq_iterator:
        try:
            metasource_variational_update_from_events(gp_type, scenes, optimizers, schedulers, metrics)
        except Exception as e:
            print("~~Inference crashed!~~")
            traceback.print_exc()
            return np.nan

        # Log various metrics and plot scenes
        if step_idx % config["report"]["print"] == 0:
            taq_iterator.set_description("Epoch {:05d} | Epoch loss {:.1f}".format(step_idx, metrics.epoch_loss[-1] ))

        if step_idx % config["report"]["plot"] == 0:
            metrics.plot(savepath)

    print("Inference complete!")
    return metrics.epoch_loss[-1]


def metasource_variational_update_from_events(gp_type, scenes, optimizers, schedulers, metrics, chunk_size=13):
    """ One iteration of sampling from Scenes conditioned on
        event-level variables, and apply gradient updates to
        learn meta-source parameters
    """

    # Shuffle which scenes are optimized together
    np.random.shuffle(scenes.scene_assignments)  # will only shuffle along first dimension
    if scenes.scene_assignments.shape[1] == 1:  # only one device
        chunks = [
            scenes.scene_assignments[x:x+chunk_size, 0] for x in range(0, len(scenes.scene_assignments), chunk_size)
            ]
    else:
        chunks = [list(scenes.scene_assignments[j, :]) for j in range(scenes.scene_assignments.shape[0])]

    # Optimization split into chunks to make memory usage manageable
    for scene_idxs in chunks:
        # Run forward and backward
        for optimizer in optimizers:
            optimizer.zero_grad()
        scores = scenes.event_forward(gp_type, scene_idxs)
        loss = inference.metrics.mean_score(scores)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        metrics.step_update(loss, scores)

    # Update scheduler only after all the scenes have been seen
    metrics.epoch_update(scenes, len(chunks))
    if schedulers is not None:
        schedule_cycle = context.optimization["schedule_cycle"]
        if (len(metrics.epoch_loss) % schedule_cycle == 0) and (len(metrics.epoch_loss) >= schedule_cycle):
            for scheduler in schedulers:
                scheduler.step(metrics.epoch_loss[-1])
