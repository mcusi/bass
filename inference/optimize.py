import sys
import time
import numpy as np
import traceback
from datetime import datetime

import torch
import torch.optim

import inference.io
import inference.metrics
from util.sample import sample_delta
from util.context import context

################
# Basic inference loops (single scene)
################

def variational_update(scene, optimizers, schedulers, metrics, accumulate_gradients=False):
    """ One iteration of sampling from Scene and applying gradient updates """
    start_time = time.time()

    # Run forward and backward
    for optimizer in optimizers:
        optimizer.zero_grad()

    if accumulate_gradients:
        # Option for inference that is robust to running out of memory
        # To accumulate gradients, split a single iteration of optimization into two.
        # Limit at 2 iters for time efficiency
        for accumlation_idx in range(2):
            scores = scene.pq_forward()
            loss = inference.metrics.importance_weighted_bound(scores)/2
            loss.backward()
            metrics.accumulate(scene, loss, scores, accumlation_idx)
    else:
        if context.optimization['loss_type'] == "elbo":
            scores = scene.pq_forward()
            loss = inference.metrics.importance_weighted_bound(scores)
        elif context.optimization['loss_type'] == "map":
            with context(batch_size=1):
                scene.pq_forward(sample=sample_delta)
                scores = scene.ll + scene.lp
                loss = -scores.mean()
        elif context.optimization['loss_type'] == "ml":
            with context(batch_size=1):
                scene.pq_forward(sample=sample_delta)
                scores = scene.ll
                loss = -scores.mean()
        loss.backward()

    for optimizer in optimizers:
        optimizer.step()

    # Update metrics and optimizers
    if accumulate_gradients:
        metrics.accumulation_update(scene, step_time=time.time()-start_time)
    else:
        metrics.update(scene, loss, scores, time.time()-start_time)
    if schedulers is not None:
        schedule_cycle = context.optimization["schedule_cycle"]
        if (len(metrics.loss) % schedule_cycle == 0) and (len(metrics.loss) >= schedule_cycle):
            for scheduler in schedulers:
                scheduler.step(np.mean(metrics.loss[-schedule_cycle:]))


def basic_loop_from_scene(scene, optimizers, schedulers, metrics, config, savepath="./", earlystop=False, round_type=None, accumulate_gradients=False):
    """Perform inference starting from a Scene object. Requires `with context(...):` at level above."""
    # Set up number of steps to perform during this phase of inference
    scene_device = scene.n_sources.device
    if earlystop is True:
        if "cleanup" in round_type:
            last_step = config["heuristics"]["cleanup"]["early_stop_steps"]
        else:
            last_step = config["heuristics"]["earlystop"]["steps"]
    elif round_type is not None and (("cleanup" in round_type) or (round_type == "sequential_final")):
        last_step = config["heuristics"]["sequential"]["full_duration_steps"]
    else:
        last_step = config["optimization"]["total_steps"]
    taq_iterator = range(len(metrics.loss), last_step+1)

    # Get an initial look at the scene before starting optimization
    print("CUDA Memory Allocated: ", torch.cuda.memory_allocated()/1e9)
    if len(metrics.loss) == 0:
        with torch.no_grad():
            scene.pq_forward()
            scene.plot(savepath, -1)
    # Toggle accumulate gradients to avoid memory errors
    if accumulate_gradients:
        print("Using gradient accumulation!")
        # To accumulate gradients, split a single iteration of optimization into two.
        context.batch_size = int(context.batch_size/2)

    print("~~~~~Starting inference from step ", len(metrics.loss))
    #S tart optimization loop
    for step_idx in taq_iterator:
        try:
            variational_update(scene, optimizers, schedulers, metrics, accumulate_gradients=accumulate_gradients)
        except Exception as e:
            print("~~Inference crashed!~~")
            print(str(e),flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            if "cholesky" in str(e).lower() or "positive definite" in str(e).lower():
                print("Cholesky error.")
                print("~~Returning nan on device {}...".format(scene_device),flush=True)
                return np.nan
            elif "cuda" in str(e).lower() or "cufft" in str(e).lower():
                # Memory error on GPU
                print("Cuda error")
                if (not accumulate_gradients) and (context.optimization['loss_type'] == "elbo"):
                    accumulate_gradients = True
                    context.batch_size = int(context.batch_size/2)
                    print("Trying gradient accumulation once, decrease batch size to ", context.batch_size) 
                    continue
                else:
                    print("~~Crashing to attempt robust run...".format(scene_device), flush=True)
                    sys.exit(1)
            else:
                print("~~Returning nan on device {}...".format(scene_device), flush=True)
                return np.nan

        # Log various metrics and plot scene
        if (step_idx % config["report"]["print"] == 0):
            print("{} | step {:05d} | loss {:.6f} | ELBO {:.1f} | ESS {:.3f} | dev {}".format(datetime.now(), step_idx, np.mean(metrics.loss[-20:]), metrics.elbo(), metrics.ess(), scene_device), flush=True)

        if (step_idx % config["report"]["ckpt"] == 0) or (step_idx == last_step):
            inference.io.save_state(savepath, scene, step_idx, metrics, optimizers, schedulers)         
            inference.io.save_metrics(savepath, metrics)

        if (step_idx % config["report"]["plot"] == 0) or (step_idx == last_step):
            metrics.plot(savepath)
            with torch.no_grad():
                scene.plot(savepath, step_idx)

    print("Inference complete!")
    importance_sampled_elbo = metrics.elbo()
    return importance_sampled_elbo


def early_stop_from_scene(scene, optimizers, schedulers, metrics, config, savepath="./"):
    """Like basic_loop_from_scene, but using fewer steps"""
    success = basic_loop_from_scene(scene, optimizers, schedulers, metrics, config, savepath, earlystop=True)
    if np.isnan(success):
        return np.nan
    else:
        return (-np.array(metrics.loss)).tolist()
