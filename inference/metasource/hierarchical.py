import yaml
import os

import inference.io
import inference.proposals
from inference.serial import hypothesis_setup
import inference.metasource.optimize
from model.scenes import Scenes
from util.sample import manual_seed
from util.context import context


def run(hypotheses, observations, observation_names, audio_sr, dataset_name, expt_name, config_name, seed=None):
    """Infer variational posterior including meta-source parameters, by conditioning on audio"""

    manual_seed(seed)
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
    devices = inference.io.get_device("all")

    with context(experiment_config, batch_size=experiment_config["optimization"]["batch_size"]):
        scenes_init = Scenes.hypothesize(hypotheses, observations, observation_names, audio_sr, experiment_config, devices)

    scenes_opt, optimizers, schedulers, metrics, hypothesis_config, savepath = hypothesis_setup(dataset_name, "hierarchical", expt_name, hypothesis_name="h", seed=seed, experiment_config=experiment_config, scene_init=scenes_init, device=None, hierarchical=True)
    with context(hypothesis_config, batch_size=hypothesis_config["optimization"]["batch_size"]):
        elbo_is = inference.metasource.optimize.infer(scenes_opt, optimizers, schedulers, metrics, hypothesis_config, savepath=savepath)

    return scenes_opt, elbo_is, savepath


def symbolic_run(gp_type, hypotheses, observations, observation_names, audio_sr, dataset_name, expt_name, config_name, seed=None):
    """Infer variational posterior including meta-source parameters, by conditioning on event variables"""

    manual_seed(seed)
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
    devices = inference.io.get_device("all")

    with context(experiment_config, batch_size=experiment_config["optimization"]["batch_size"]):
        scenes_init = Scenes.hypothesize(hypotheses, observations, observation_names, audio_sr, experiment_config, devices)

    scenes_opt, optimizers, schedulers, metrics, hypothesis_config, savepath = hypothesis_setup(dataset_name, "hierarchical", expt_name, hypothesis_name="h", seed=seed, experiment_config=experiment_config, scene_init=scenes_init, device=None, hierarchical=True)
    with context(hypothesis_config, batch_size=hypothesis_config["optimization"]["batch_size"]):
        elbo_is = inference.metasource.optimize.infer_from_events(gp_type, scenes_opt, optimizers, schedulers, metrics, hypothesis_config, savepath=savepath)

    return scenes_opt, elbo_is, savepath
