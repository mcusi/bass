#!/bin/bash
# The following line allows this file to be run directly as a slurm script (e.g. "sbatch distributed.py")
# -----------------------
# ''''python -u -- ${home_dir}/inference/distributed.py ${1+"$@"}; exit; # '''
''''python -u -- ${home_dir}/inference/distributed.py ${1+"$@"} || python -u -- ${home_dir}/inference/distributed.py --robust ${1+"$@"}; python -u -- ${home_dir}/inference/distributed.py --end ${1+"$@"}; exit; # '''
# -----------------------

import argparse
import numpy as np
from glob import glob
import json
import yaml
import os
import inspect
import shutil
import re
import subprocess
import time

import torch

import renderer.util
import inference.io
import inference.sequential_heuristics
import inference.proposals
import inference.optimize
from inference.serial import hypothesis_setup
from inference.viz import chain_viz
import inference.cleanup
from inference.construction import get_hypothesis_name, analyze_previous_round, create_new_scene_hypotheses, create_scene_hypotheses_cleanup
from util.sample import manual_seed
from util.cuda import get_exclude_nodes
from util.context import context

""" Highest-level methods for running sequential inference distributed on a SLURM cluster """


def create_scene_hypotheses_sequential(sound_name, sound_group, expt_name, experiment_config, segmentation, round_idx):
    """ Create scene hypothesis from amortized inference event proposals """
    hypotheses_dir = os.path.join(
        os.environ["inference_dir"],
        expt_name,
        sound_group,
        sound_name,
        ""
        )

    # 1. Create event proposals and observation.
    # If there aren't any left, inference is done!
    event_proposal_sequence, observation, audio_sr, full_scene_duration, last_round_of_sequential = get_event_proposals_and_observation(round_idx, sound_name, sound_group, segmentation, experiment_config)
    if observation is None and audio_sr is None:
        return None, None  # Finished

    # 2. Analyze previous round (save results.json etc.)
    # Hypotheses will be called "round00X-0XX" 
    if round_idx == 0:
        os.makedirs(hypotheses_dir, exist_ok=True)
    keep_idxs = [] if round_idx == 0 else analyze_previous_round(hypotheses_dir, round_idx)

    # 3. Propose hypotheses for current round
    scene_hypotheses = create_new_scene_hypotheses(
        event_proposal_sequence,
        keep_idxs,
        round_idx,
        experiment_config,
        observation,
        audio_sr,
        hypotheses_dir
        )

    # 3. Filter hypotheses using sequential_heuristics
    if experiment_config["heuristics"]["sequential"]["use"] is True:
        scene_hypotheses = inference.sequential_heuristics.prioritize(
            scene_hypotheses, experiment_config, round_idx=round_idx
            )
    # 3b. Add background only on round_idx == 0
    # if desired - might double the number of proposals
    if experiment_config["optimization"]["add_background_noise_proposal"] and round_idx == 0:
        # Proposal gets appended to scene_hypotheses.
        inference.proposals.add_background_proposal(
            scene_hypotheses, round_idx, experiment_config,
            observation, audio_sr, full_scene_duration
            )

    return scene_hypotheses, last_round_of_sequential


def iterative(sound_name, sound_group, expt_name, config_name, segmentation, round_type, round_idx=0, seed=None):
    """ Run inference for a scene using amortized event proposals
        by submitting each hypothesis as a new slurm job
    """

    # Set up config for event proposals
    hypotheses_dir = os.path.join(
        os.environ["inference_dir"], expt_name, sound_group, sound_name, ""
        )
    manual_seed(seed)
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    # 1. Create scene hypotheses
    if "sequential" in round_type:
        scene_hypotheses, last_round_of_sequential = create_scene_hypotheses_sequential(
            sound_name, sound_group, expt_name,
            experiment_config, segmentation, round_idx
            )
        if (scene_hypotheses is None) and (last_round_of_sequential is None):
            # Inference is done.
            slurm_launch(on_complete, cpu=True,
                         job_name="complete",
                         log_dir=f"{hypotheses_dir}/slurm_logs"
                         )()
            return
        if last_round_of_sequential:
            round_type = round_type + "_final"
    elif "cleanup" in round_type:
        scene_hypotheses, original_savepaths = create_scene_hypotheses_cleanup(
            sound_name, sound_group, expt_name,
            experiment_config, round_type, round_idx
            )

    # 4. Create folders for all the hypotheses
    for hypothesis_number, scene_hypothesis in enumerate(scene_hypotheses):
        hypothesis_name = get_hypothesis_name(
            round_idx, hypothesis_number, round_type
            )
        savepath = inference.io.get_directory(
            sound_name, sound_group, expt_name,
            hypothesis_name=hypothesis_name, seed=None
            )
        if "cleanup" in round_type and hypothesis_number < experiment_config["optimization"]["n_to_keep"]:
            # Single out the "originals" upon which the other scenes are based
            if not experiment_config["heuristics"]["cleanup"]["optimize_originals"]:
                # Copy old directory into new directory
                src = original_savepaths[hypothesis_number]
                dst = savepath
                shutil.copytree(src, dst, dirs_exist_ok=True)
                continue
        # Otherwise just go on and save the hypothesis as usual.
        inference.io.get_config(savepath, experiment_config=experiment_config)
        inference.io.save_hypothesis(savepath, scene_hypothesis)

    # 5. SBATCH --array=... optimize
    earlystop = experiment_config['heuristics']['earlystop']['use'] if "sequential" in round_type else experiment_config['heuristics']['cleanup']['use_earlystop']
    job_array_id = slurm_launch(
        solo_optimization,
        array=range(len(scene_hypotheses)),
        job_name=f"R{round_idx}",
        log_dir=f"{hypotheses_dir}/slurm_logs",
        return_jobid=True)(earlystop=earlystop, round_type=round_type)

    # 6. if early_stop, figure out which ones to continue.
    # otherwise, analyze results of this round.
    if earlystop:
        slurm_launch(
            continue_after_early_stop,
            cpu=True,
            dependency=f"afterany:{job_array_id}",
            job_name=f"R{round_idx}continue",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )(round_idx=round_idx, round_type=round_type)
    else:
        # analyze_complete_round will also launch viz
        # and next iteration=round_idx+1
        slurm_launch(
            analyze_complete_round,
            cpu=True,
            dependency=f"afterany:{job_array_id}",
            job_name=f"R{round_idx}complete",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )(round_idx=round_idx, round_type=round_type)

    return


def trim_observation(full_observation, audio_sr, config, scene_offset=-1):
    """Trim and apply a ramp to prevent artefacts for sequential inference."""
    copied_full_observation = full_observation.copy()
    if scene_offset > 0:
        scene_offset_idx = int(np.floor(audio_sr * scene_offset))
        obs = copied_full_observation[np.newaxis, :scene_offset_idx]
        off_ramp = renderer.util.hann_ramp(
            audio_sr, ramp_duration = config["renderer"]["ramp_duration"]
            )
        # prevents splicing artifact
        obs[:, -off_ramp.shape[0]:] = obs[:, -off_ramp.shape[0]:]*off_ramp
    else:
        obs = copied_full_observation[np.newaxis, :]
    return obs


def get_event_proposals_and_observation(round_idx, sound_name, sound_group, segmentation, experiment_config):
    """ Get event proposals and observation pertaining to this round """

    # Load information from amortized inference, including properly sampled observation 
    amortize = {'segmentation': segmentation}
    event_proposals, full_observation, audio_sr = inference.proposals.load_event_proposals_and_sound(
        sound_name, sound_group, amortize,
        source_types_to_keep=experiment_config["hyperpriors"]["source_type"]["args"],
        minimum_duration=experiment_config["renderer"]["steps"]["t"]
        )
    full_scene_duration = len(full_observation)/audio_sr

    # Group event proposals based on their temporal sequence
    event_proposal_sequence = inference.proposals.group_event_proposals(
        event_proposals, experiment_config
        )
    if len(event_proposal_sequence) == 0:
        print("No events in the event_proposal_sequence.")
        return [None] * 5
    # Filter event proposals to only the ones that are on this round
    new_event_proposals = [
        (es["events"][0]["meta"]["rank"], es)
        for es in event_proposal_sequence
        if es["events"][0]["meta"]["round_idx"] == round_idx
        ]

    # If round_idx is the max of the round_idxs, then we are on the last round
    # and we will look at the entire observation
    last_round = np.max([
        es["events"][0]["meta"]["round_idx"]
        for es in event_proposal_sequence
        ])
    if round_idx > last_round:
        print("Inference is complete.")
        return [None] * 5
    scene_offset = -1 if round_idx == last_round else [es["events"][0]["meta"]["group_onset"] for es in event_proposal_sequence if es["events"][0]["meta"]["round_idx"] == round_idx + 1][0]

    # Trim the observation to the current section for inference
    observation = trim_observation(
        full_observation,
        audio_sr,
        experiment_config,
        scene_offset=scene_offset
        )
    last_round_of_sequential = False
    if scene_offset == -1:
        scene_offset = full_scene_duration
        last_round_of_sequential = True

    return new_event_proposals, observation, audio_sr, full_scene_duration, last_round_of_sequential


def continue_after_early_stop(sound_name, sound_group, expt_name, config_name, round_type, round_idx):
    """ Resubmit slurm jobs per hypothesis if they pass the earlystop check """

    # Load in config and location of early stop results
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
    hypotheses_dir = os.path.join(
        os.environ["inference_dir"], expt_name,
        sound_group, sound_name, ""
        )

    # 1. Read in all metrics for this round
    # Load results of previous hypotheses
    previous_hypothesis_folders = glob(os.path.join(
        hypotheses_dir,
        get_hypothesis_name(round_idx, "*", round_type),
        ""
        ))
    last_round_elbos = {}
    failed_hypotheses = []
    for ph_folder in previous_hypothesis_folders:
        hypothesis_number = int(
            re.match(".*/round[0-9]*-([0-9]*)", ph_folder).group(1)
            )
        try:
            with open(ph_folder + "state.txt", "r") as f:
                # if state.txt exists, it should read "earlystop"
                hypothesis_status = f.readlines()
        except:
            # if state.txt doesn't exist, then slurm crashed
            failed_hypotheses.append(hypothesis_number)
        try:
            with open(ph_folder + "metrics.json", "r") as f:
                metrics = json.load(f)
            last_round_elbos[hypothesis_number] = metrics["elbo"]
        except:
            last_round_elbos[hypothesis_number] = -np.inf

    if len(failed_hypotheses) > 0:
        # Need to resubmit those earlystop that
        # crashed because of slurm, then recheck which won
        print("slurm-Failed hypotheses (resubmitting): ", failed_hypotheses)
        job_array_id = slurm_launch(
            solo_optimization,
            array=failed_hypotheses,
            job_name=f"R{round_idx}",
            log_dir=f"{hypotheses_dir}/slurm_logs",
            return_jobid=True
            )(earlystop=True, round_type=round_type)
        slurm_launch(
            continue_after_early_stop,
            cpu=True,
            dependency=f"afterany:{job_array_id}",
            job_name=f"R{round_idx}continue",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )(round_idx=round_idx, round_type=round_type)
    else:
        # 2. Figure out what did best, filter out the worse ones
        ordered_hypotheses = sorted(
            last_round_elbos.keys(),
            key=lambda k: last_round_elbos[k],
            reverse=True
            )
        ordered_hypotheses = [
            oh for oh in ordered_hypotheses
            if not np.isinf(last_round_elbos[oh])
            ]
        n_to_keep = experiment_config["heuristics"]["earlystop"]["cutoff"]
        keep_idxs = ordered_hypotheses[:n_to_keep]

        # Update the state of each hypothesis
        for ph_folder in previous_hypothesis_folders:
            hypothesis_number = int(re.match(".*/round[0-9]*-([0-9]*)", ph_folder).group(1))
            with open(ph_folder + "state.txt", "w") as f:
                if hypothesis_number in keep_idxs:
                    f.write("continue")
                else:
                    f.write("complete")

        # 3. SBATCH --array=... optimize
        job_array_id = slurm_launch(
            solo_optimization,
            array=keep_idxs,
            return_jobid=True,
            job_name=f"R{round_idx}continue",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )()

        # 4. SBATCH --dependency=after complete(round_idx)
        slurm_launch(
            analyze_complete_round,
            cpu=True,
            dependency=f"afterany:{job_array_id}",
            job_name=f"R{round_idx}complete",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )(round_idx=round_idx)


def analyze_complete_round(sound_name, sound_group, expt_name, config_name, round_type, round_idx):
    """ Analyze complete round=round_idx of iterative inference
        to pass on the top N hypotheses to round=round_idx+1
    """

    # Set up config for event proposals
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    # Load results of previous hypotheses
    hypotheses_dir = os.path.join(
        os.environ["inference_dir"], expt_name, sound_group, sound_name, ""
        )
    previous_hypothesis_folders = glob(os.path.join(
        hypotheses_dir, 
        get_hypothesis_name(round_idx, "*", round_type),
        ""
        ))

    last_round_elbos = {}
    failed_hypotheses = []
    for ph_folder in previous_hypothesis_folders:
        hypothesis_number = int(
            re.match(".*/round[0-9]*-([0-9]*)", ph_folder).group(1)
            )
        with open(os.path.join(ph_folder, "state.txt"), "r") as f:
            hypothesis_status = f.readlines()
            print(f"{hypothesis_number}: {hypothesis_status}")
        # options: continue, earlystop, complete
        if "".join(hypothesis_status) == "continue":
            failed_hypotheses.append(hypothesis_number)
        try:
            with open(ph_folder + "metrics.json", "r") as f:
                metrics = json.load(f)
            last_round_elbos[hypothesis_number] = metrics["elbo"]
        except:
            last_round_elbos[hypothesis_number] = -np.inf

    if len(failed_hypotheses) > 0:
        print("slurm-Failed hypotheses (resubmitting): ", failed_hypotheses)
        # If slurm failed on any hypotheses, resubmit
        # those hypotheses and then redo analyze_complete_round
        job_array_id = slurm_launch(
            solo_optimization,
            array=failed_hypotheses,
            job_name=f"R{round_idx}",
            log_dir=f"{hypotheses_dir}/slurm_logs",
            return_jobid=True
            )(earlystop=False, round_type=round_type)
        slurm_launch(
            analyze_complete_round,
            cpu=True,
            dependency=f"afterany:{job_array_id}",
            job_name=f"R{round_idx}complete",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )(round_idx=round_idx, round_type=round_type)
    else:
        # Figure out what did best
        ordered_hypotheses = sorted(
            last_round_elbos.keys(),
            key=lambda k: last_round_elbos[k],
            reverse=True
            )
        n_to_keep = experiment_config["optimization"]["n_to_keep"]
        keep_idxs = ordered_hypotheses[:n_to_keep]
        reject_idxs = ordered_hypotheses[n_to_keep:]
        print(f"On round {round_idx}, {round_type}: keep=", keep_idxs,
              ", reject=", reject_idxs,
              flush=True)

        # Save results
        results_fn = os.path.join(hypotheses_dir, "results.json")
        if os.path.isfile(results_fn):
            with open(results_fn, 'r') as outfile:
                saved_results = json.load(outfile)
            n_rounds_in_saved_results = len(saved_results["keep"])
            if n_rounds_in_saved_results == round_idx:
                saved_results["keep"].append(keep_idxs)
                saved_results["reject"].append(reject_idxs)
                with open(results_fn, 'w') as outfile:
                    json.dump(saved_results, outfile)
            elif saved_results["keep"][round_idx] == keep_idxs and saved_results['reject'][round_idx] == reject_idxs:
                print(f"results.json already has correct \
                      indices for round {round_idx}. Skipping save.")
            else:
                raise Exception("results.json is inconsistent with current run!")

        else:
            results = {"keep": [keep_idxs], "reject": [reject_idxs]}
            with open(results_fn, 'w') as outfile:
                json.dump(results, outfile)

        # Save round type
        round_type_fn = os.path.join(hypotheses_dir, "roundtype.txt")
        round_type_f = open(round_type_fn, "a")
        round_type_f.write(f"{round_idx}:{round_type}\n")
        round_type_f.close()

        # Launch viz for this complete round
        slurm_launch(
            viz,
            cpu=True,
            job_name=f"R{round_idx}viz",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )()

        # Launch next round
        if ("cleanup" in round_type) or (round_type == "sequential_final"):
            round_type = inference.cleanup.get_next_round_type(
                round_type,
                keep_idxs=keep_idxs,
                use_cleanup_flag=experiment_config["heuristics"]["cleanup"].get("use", True)
                )
            if round_type is None:
                print("Inference done, we made it! Exiting.")
                slurm_launch(
                    on_complete,
                    cpu=True,
                    job_name="complete",
                    log_dir=f"{hypotheses_dir}/slurm_logs"
                    )()
                return

        slurm_launch(
            iterative,
            cpu=True,
            mem=100,
            job_name=f"R{round_idx+1}",
            log_dir=f"{hypotheses_dir}/slurm_logs"
            )(round_type=round_type, round_idx=round_idx+1)


def solo_optimization(sound_name, sound_group, expt_name, round_type, round_idx, hypothesis_number, robust, earlystop=False, seed=None, cuda_idx=0):
    """ Infer variational posterior for a hypothesis
        which already has an initialization saved in a folder
    """
    manual_seed(seed)
    print("Optimization with robust: ", robust)
    device = inference.io.get_device(cuda_idx) if torch.cuda.is_available() else "cpu"
    hypothesis_name = get_hypothesis_name(round_idx, hypothesis_number)
    scene_opt, optimizers, schedulers, metrics, hypothesis_config, savepath = hypothesis_setup(
        sound_name, sound_group, expt_name,
        hypothesis_name, None, None, None, device
        )
    with context(hypothesis_config, batch_size=hypothesis_config["optimization"]["batch_size"]):
        elbo_is = inference.optimize.basic_loop_from_scene(
            scene_opt,
            optimizers,
            schedulers,
            metrics,
            hypothesis_config,
            savepath=savepath,
            earlystop=earlystop,
            round_type=round_type,
            accumulate_gradients=robust
            )
    print("ELBO_is: ", elbo_is)
    return elbo_is


def save_solo_optimization(sound_name, sound_group, expt_name, round_type, round_idx, hypothesis_number, robust, earlystop=False, seed=None, cuda_idx=0):
    """ Save a file which proves that slurm did not crash;
        rather, optimization finished or it crashed because of cuda
    """
    hypothesis_name = get_hypothesis_name(round_idx, hypothesis_number)
    savepath = inference.io.get_directory(
        sound_name, sound_group, expt_name,
        hypothesis_name=hypothesis_name, seed=None
        )
    with open(savepath + "state.txt", "w") as f:
        if earlystop:
            f.write("earlystop")
        else:
            f.write("complete")
    return


def slurm_launch(f, cpu=None, array=None, n_mins=150, job_name=None, log_dir=None, dependency=None, mem=32, return_jobid=False, confirm=False):
    # Run like:
    # slurm_launch(f, ...)(**f_kwargs)
    def wrapped_f(**kwargs):
        sbatch_args = {
            **SLURM_ARGS,
            "time": "{}:00".format(n_mins),
            "mem": f"{mem}G"
        }
        if cpu == True:
            sbatch_args['gres'] = None
            sbatch_args['constraint'] = None
        sbatch_args['exclude'] = get_exclude_nodes(
            sbatch_args.get('gres', None)
            )

        py_args = {**PY_ARGS, **kwargs, "command": f.__name__}

        if job_name is not None:
            sbatch_args["job-name"] = job_name
        if array is not None:
            sbatch_args["array"] = ",".join(map(str, array))
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            suffix = f"_{job_name}" if job_name is not None else ""
            fname = f"slurm-%j{suffix}.out" if array is None else f"slurm-%A_%a{suffix}.out"
            sbatch_args["output"] = f"{log_dir}/{fname}"
            sbatch_args["error"] = f"{log_dir}/{fname}"
        if dependency is not None:
            sbatch_args["dependency"] = dependency

        cmd = " ".join([
            "sbatch",
            "--parsable",
            *[f"--{k}={v}" for k, v in sbatch_args.items() if v is not None],
            __file__,
            *[f"--{k}={v}" for k, v in py_args.items() if v is not None and not isinstance(v, bool)],
            *[f"--{k}" for k, v in py_args.items() if v is True]
        ])

        if confirm:
            print("\nAre you sure you want to run the following command:\n")
            print(cmd)
            if input("\n[y/n]? ") not in ["y", "Y", ""]: return
        else:
            print("\nRunning the following sbatch command:\n")
            print(cmd)

        if return_jobid:
            while True:
                jobid = subprocess.run(cmd.split(" "), capture_output=True).stdout.strip()
                if jobid:
                    print("Job id:", jobid)
                    return int(jobid)
                else:
                    print("Failed to get sbatch job id, retrying in 60s...")
                    time.sleep(60)
        else:
            os.system(cmd)
    return wrapped_f


def viz(sound_name, sound_group, expt_name):
    chain_viz(expt_name, sound_group, sound_name)


def _launch_multiple(*sound_names):
    sound_name = sound_names[0]
    sound_queue = ",".join(sound_names[1:]) if len(sound_names) > 1 else None
    slurm_launch(iterative)(
        sound_name=sound_name,
        sound_queue=sound_queue,
        round_idx=0,
        round_type="sequential"
        )


def on_complete(sound_queue):
    if sound_queue is not None:
        _launch_multiple(*sound_queue.split(","))


def sweep(sound_group, sound_queue):
    if sound_queue is None:
        wavs = glob(os.path.join(os.environ["sound_dir"], sound_group, "*.wav"))
        _launch_multiple(*[
            os.path.splitext(os.path.basename(w))[0] for w in wavs
            ])
    else:
        _launch_multiple(*sound_queue.split(","))


def sbatch(sound_group, sound_name, sound_queue):
    if sound_name is None:
        sweep(sound_group, sound_queue)
    else:
        slurm_launch(iterative)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Which function to run
    parser.add_argument("--command", type=str, default="iterative")
    parser.add_argument("--end", action="store_true")
    # Experiment args
    parser.add_argument("--sound-name", type=str, default=None,
                        help="name of wav file")
    parser.add_argument("--sound-group", type=str,
                        help="folder where wavs are saved")
    parser.add_argument("--expt-name", type=str,
                        help="name of inference folder")
    parser.add_argument("--config-name", type=str,
                        help="name of inference config")
    parser.add_argument("--segmentation", type=str,
                        help="name of neural network for amortized")
    parser.add_argument("--seed", type=int, default=None)
    # Function args
    parser.add_argument("--round-idx", type=int, default=0,
                        help="index of sequential inference round")
    parser.add_argument("--round-type", type=str, default="sequential",
                        help="type of inference proposal in this round")
    parser.add_argument("--earlystop", action="store_true",
                        help="run fewer iters of variational inference?")
    # Slurm args
    parser.add_argument("--slurm.partition", type=str, default=os.environ['slurm_partition'])
    parser.add_argument("--slurm.gres", type=str, default=None)
    parser.add_argument("--slurm.constraint", type=str, default=None)
    parser.add_argument("--robust", action="store_true")
    # Queue
    parser.add_argument("--sound-queue", type=str, default=None,
                        help="other sound names to be submitted")

    args = parser.parse_args()
    print(args, flush=True)

    if args.seed is not None:
        assert f"seed{args.seed}" in args.expt_name

    f_dict = {
        'iterative': iterative,
        'continue_after_early_stop': continue_after_early_stop,
        'solo_optimization': solo_optimization,
        'viz': viz,
        'sweep': sweep,
        'analyze_complete_round': analyze_complete_round,
        'on_complete': on_complete,
        'sbatch': sbatch
        }
    f = f_dict[args.command]

    PY_ARGS = vars(args)
    SLURM_ARGS = {k.replace("slurm.", ""):v for k, v in vars(args).items()
                  if k.startswith("slurm.") and v is not None}

    f_args = {k: v for k, v in PY_ARGS.items()
              if k in inspect.getfullargspec(f)[0]}
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        f_args['hypothesis_number'] = int(os.environ['SLURM_ARRAY_TASK_ID'])

    print("Running:")
    print(" ".join([
        "python",
        __file__, 
        *[f"--{k}={v}" for k, v in PY_ARGS.items() if v is not None and not isinstance(v, bool)],
        *[f"--{k}" for k, v in PY_ARGS.items() if v is True]
    ]))
    print()

    if args.end:
        if args.command == "solo_optimization":
            save_solo_optimization(**f_args)
    else:
        f(**f_args)
