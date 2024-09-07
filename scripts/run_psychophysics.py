import argparse
import os
import inference.serial
import numpy as np
import subprocess
from util import get_exclude_nodes


def run_experiment(expt_type, sound_group, config_name, expt_name, expt_idx=None, seed=0, n_seeds=None, partition=None, device=None, start_with_seed=None):
    # Select the experiment
    full_expt_name = "_".join([expt_name, config_name])
    print(full_expt_name)
    if expt_type == "perceptual_induction":
        import psychophysics.hypotheses.perceptual_induction as hyp
        overwrite = {"seed": seed}
    elif expt_type == "spectral_completion":
        # Running the spectral completion experiment
        import psychophysics.hypotheses.spectral_completion as hyp
        overwrite = {"seed": seed}
    elif expt_type == "spectral_completion_match":
        # Generating the latents for the spectral_completion comparison stimuli
        import psychophysics.hypotheses.spectral_completion_match as hyp
        overwrite = {"seed": seed}
    elif expt_type == "mistuned_harmonic":
        import psychophysics.hypotheses.mistuned_harmonic as hyp
        overwrite = {}
    elif expt_type == "onset_asynchrony":
        import psychophysics.hypotheses.onset_asynchrony as hyp
        overwrite = {"vowelDuration": 60, "offsets": [0.], "seed": seed}
    elif expt_type == "cmr":
        import psychophysics.hypotheses.cmr as hyp
        # Default overwrite of 10 Hz for the lowpass modulation frequency,
        # because of our input representation
        overwrite = {
                        "n_trials": 1,
                        "lowpass_for_mod": 10,
                        "tone_levels": np.arange(40, 90, 5),
                        "bandwidths": [1000, 400, 200, 100],
                        "seed": seed
                    }
    elif expt_type == "tone_sequences":
        import psychophysics.hypotheses.tone_sequences as hyp
        overwrite = {
                        "bouncing": {},
                        "compete": {},
                        "aba": {},
                        "cumulative": {},
                        "captor": {"tone_duration": 0.070}
                    }
    else:
        raise NotImplementedError("expt_type=" + expt_type)
    
    # Create sounds & create cache, and collect initializations
    experiments = hyp.full_design(
        sound_group, config_name, overwrite=overwrite
        )

    # Slurm launching
    if expt_idx is None:
        # Launch inference for this seed
        if n_seeds is None or seed == 0:
            next_seed = 0 if start_with_seed is None else start_with_seed
            py_args = {
                "expt_type": expt_type,
                "sound_group": sound_group,
                "config_name": config_name,
                "expt_name": expt_name,
                "seed": next_seed,
                "n_seeds": n_seeds,
                "partition": partition,
                "device": device
                }
            slurm_launch(
                py_args, start_id=0, end_id=len(experiments) - 1,
                partition=partition, device=device
                )

    else:
        if n_seeds is not None and seed < n_seeds-1 and expt_idx == len(experiments)-1:
            next_seed = seed+1
            py_args = {
                "expt_type": expt_type,
                "sound_group": sound_group,
                "config_name": config_name,
                "expt_name": expt_name,
                "seed": next_seed,
                "n_seeds": n_seeds,
                "partition": partition,
                "device":device
                }
            slurm_launch(
                py_args, start_id=0, end_id=len(experiments) - 1,
                partition=partition, device=device
                )

        # Select the hypothesis/initialization
        experiment_key = [k for k in experiments.keys()][expt_idx]
        sound_name = experiment_key[0]
        hypothesis_name = experiment_key[1]
        hypothesis = experiments[experiment_key]
        print(experiment_key, flush=True)
        # Run inference
        inference.serial.solo_optimization(hypothesis, sound_name, sound_group, full_expt_name, hypothesis_name, config_name, seed=seed)


def slurm_launch(py_args, start_id=None, end_id=None, n_hours=2, partition=os.environ["slurm_partition"], device=None, log_dir=None, max_concurrent=None, dependency=None, return_jobid=False):

    if max_concurrent is None:
        max_concurrent = 32
    sbatch_args = {
        "gres": "gpu:1",
        "constraint": "high-capacity",
        "time": "{:02d}:00:00".format(n_hours),
        "mem": "6G",
        "partition": f"{partition or os.environ['slurm_partition']}",
        "exclude": get_exclude_nodes(device)
    }
    if device == 'gpu':
        sbatch_args['gres'] = 'gpu:1'
        sbatch_args['constraint'] = 'high-capacity'
    if start_id is not None and end_id is not None:
        sbatch_args["array"] = f"{start_id}-{end_id}%{max_concurrent}"
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        sbatch_args["output"] = f"{log_dir}/slurm-%j.out"
        sbatch_args["error"] = f"{log_dir}/slurm-%j.out"
    if dependency is not None:
        sbatch_args["dependency"] = dependency

    cmd = " ".join([
        "sbatch",
        "--parsable",
        *[f"--{k}={v}" for k,v in sbatch_args.items()],
        "run_psychophysics.sh",
        *[f"{k}={v}" for k,v in py_args.items() if v is not None]
    ])

    print("\nRunning the following sbatch command:\n")
    print(cmd)
    if return_jobid:
        jobid = int(subprocess.run(cmd.split(" "), capture_output=True).stdout.strip())
        print("Job id:", jobid)
        return jobid
    else:
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt-type", type=str,
                        help="which psychophysics experiment?")
    parser.add_argument("--sound-group", type=str,
                        help="folder to save wavs")
    parser.add_argument("--config-name", type=str,
                        help="name of yaml file for inference configs")
    parser.add_argument("--expt-name", type=str,
                        help="folder to group inference results together")
    parser.add_argument("--expt-idx", type=int,
                        help="typically SLURM_ARRAY_TASK_ID", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="how many seeds to run")
    parser.add_argument("--partition", type=str, default=None,
                        help="slurm partition")
    parser.add_argument("--device", type=str, default="gpu",
                        help="cpu or gpu")
    parser.add_argument("--start-with-seed", type=int, default=None,
                        help="start from non-zero seed")
    args = parser.parse_args()
    print("Running psychophysics!", flush=True)
    print(args, flush=True)

    # for backup, relaunch this exact job
    py_args = {
        "expt_type": args.expt_type,
        "sound_group": args.sound_group,
        "config_name": args.config_name,
        "expt_name": args.expt_name,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "partition": args.partition,
        "device": args.device,
        "start_with_seed": args.start_with_seed
        }
    if args.partition != "normal":
        dependency = f"afterany:{os.environ['SLURM_JOBID']}"
        job_id = slurm_launch(
            py_args,
            start_id=args.expt_idx,
            end_id=args.expt_idx,
            partition=args.partition,
            device=args.device,
            dependency=dependency,
            return_jobid=True
            )
        print("Launched backup job:", job_id, "with dependency", dependency)

    # now do the code
    run_experiment(
        args.expt_type,
        args.sound_group,
        args.config_name,
        args.expt_name,
        args.expt_idx,
        seed=args.seed,
        n_seeds=args.n_seeds,
        partition=args.partition,
        device=args.device,
        start_with_seed=args.start_with_seed
        )

    # now cancel the backup job
    if args.partition != "normal":
        cmd = f"scancel {job_id}"
        print("Cancelling backup job:", job_id)
        os.system(cmd)
