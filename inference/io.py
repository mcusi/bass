import os
import yaml
import torch
from glob import glob
import json
import dill


def get_device(cuda_idx):
    try:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        print("Cuda visible devices: ", cuda_visible_devices)
        if torch.cuda.is_available():
            if cuda_idx == "all":
                devices = ["cuda:{}".format(cvd) for cvd in cuda_visible_devices]
                return devices
            else:
                device = "cuda:{}".format(cuda_visible_devices[cuda_idx])
        else:
            device = "cpu"
    except Exception as e:
        print(e)
        print("Running on CPU")
        device = "cpu"
    return device

def get_directory(sound_name, sound_group, expt_name, hypothesis_name, seed=None, create=True):
    """
    Get directory where inference logs will be saved
    Note: for iterative inference, the run index is passed as `hypothesis_name`:int
    """

    if isinstance(hypothesis_name, int):
        hypothesis_name = "{:03d}".format(hypothesis_name)
    if seed is not None:
        # Used to test different hypotheses
        hypothesis_name = "{}-seed{}".format(hypothesis_name, seed)

    savepath = os.path.join(os.environ["inference_dir"], expt_name, sound_group, sound_name, hypothesis_name, "")

    if create and not os.path.isdir(savepath):
        print("Making directory: ", savepath)
        os.makedirs(savepath)

    return savepath


def get_config(savepath, experiment_config=None):
    """ Ensures the config in the hypothesis folder is used """
    cfile = savepath + '/config.yaml'
    if not os.path.isfile(cfile):
        print("Saving config: ", cfile)
        # Save config for completeness
        with open(cfile, 'w') as f:
            yaml.dump(experiment_config, stream=f, default_flow_style=False, sort_keys=False)
        hypothesis_config = experiment_config
    else:
        print("Loading config: ", cfile)
        with open(cfile, 'r') as f:
            hypothesis_config = yaml.load(f, Loader=yaml.FullLoader)
    return hypothesis_config


def save_hypothesis(savepath, scene_structure):
    ssfn = savepath + "/scene_structure.tar"
    torch.save(scene_structure, ssfn, pickle_module=dill)


def restore_hypothesis(savepath, scene_structure=None, checkpoint=None, device="cpu", print_this=True):
    """If a saved checkpoint exists, restore scene state from it."""
    # Will have observation of appropriate length included
    ssfn = savepath + "/scene_structure.tar"
    if scene_structure is None:
        if print_this:
            print("Loading structure checkpoint.")
        if torch.cuda.is_available():
            scene_structure = torch.load(ssfn, pickle_module=dill)
        else:
            scene_structure = torch.load(ssfn, map_location="cpu", pickle_module=dill)
    elif (scene_structure is not None) and os.path.isfile(ssfn):
        # Check that the saved and current structures are the same
        if torch.cuda.is_available():
            scene_structure = torch.load(ssfn, pickle_module=dill)
        else:
            scene_structure = torch.load(ssfn, map_location="cpu", pickle_module=dill)
    else:
        if print_this:
            print("Saving structure checkpoint.")
        torch.save(scene_structure, ssfn, pickle_module=dill)
        scene_structure = torch.load(ssfn)

    if checkpoint is None:
        checkpoint = load_latest_ckpt(savepath, device, print_this=print_this)

    if checkpoint is not None:
        if print_this:
            print("Restoring scene from checkpoint.")
        scene_structure.load_state_dict(checkpoint['scene_state_dict'])

    return scene_structure, checkpoint


def restore_optimizers(optimizers, schedulers, savepath=None, checkpoint=None, device="cpu"):
    """Restore optimizer and scheduler state from saved checkpoint.
    """
    if checkpoint is None:
        checkpoint = load_latest_ckpt(savepath, device)

    if checkpoint is not None:
        print("Restoring optimizers from checkpoint.")
        for opt_idx, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][opt_idx])
        if schedulers is not None:
            for s_idx, scheduler in enumerate(schedulers):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"][s_idx])


def load_latest_ckpt(savepath, device="cpu", print_this=True):
    """Load the latest checkpoint for a particular hypothesis"""
    optimized_checkpoints = glob(f'{savepath}/checkpoints/ckpt-*.tar')
    if len(optimized_checkpoints) == 0:
        return None
    ckptpath = max(optimized_checkpoints, key=os.path.getctime)
    if print_this:
        print('Found checkpoint: {}'.format(ckptpath))
    try:
        checkpoint = torch.load(ckptpath, map_location=device, pickle_module=dill)
    except:
        checkpoint = torch.load(ckptpath, map_location=device)
    return checkpoint


def save_state(savepath, scene, step_idx, metrics, optimizers, schedulers=None):
    """Save a checkpoint for a particular hypothesis in the process of optimization"""
    os.makedirs(f"{savepath}/checkpoints", exist_ok=True)
    torch.save({
        'step_idx': step_idx,
        'scene_state_dict': scene.state_dict(),
        'optimizer_state_dict': [ozr.state_dict() for ozr in optimizers],
        'scheduler_state_dict': [sdl.state_dict() for sdl in schedulers] if schedulers is not None else None,
        'metrics': metrics
        }, f"{savepath}/checkpoints/ckpt-{step_idx:08d}.tar", pickle_module=dill)
    delete_old_checkpoints(savepath)


def save_state_hp(savepath, scene, step_idx, metrics, optimizers, schedulers=None):
    """Save a checkpoint for a particular hypothesis in the process of optimization"""
    os.makedirs(f"{savepath}/checkpoints", exist_ok=True)
    torch.save({
        'step_idx': step_idx,
        'scene_state_dict': scene.state_dict(),
        'optimizer_state_dict': [ozr.state_dict() for ozr in optimizers],
        'scheduler_state_dict': [sdl.state_dict() for sdl in schedulers] if schedulers is not None else None,
        'metrics': metrics
        }, f"{savepath}/checkpoints/ckpt-{step_idx:08d}.tar", pickle_module=dill)


def delete_old_checkpoints(savepath):
    checkpoints = sorted(glob(f'{savepath}/checkpoints/ckpt-*.tar'), key=os.path.getctime)
    for f in checkpoints[:-1]:  # Keep the most recent one
        print("Deleting old checkpoint: ", f)
        os.remove(f)


def save_metrics(savepath, metrics):
    """Save metrics as a JSON file for easy reading"""
    with open(f"{savepath}/metrics.json", "w") as f:
        json.dump(metrics.summary_dict(), f)
