import yaml
import os

import soundfile as sf

import inference.io
import inference.optimize
import inference.optimizers
from model.scene import Scene
from util.sample import manual_seed
from util.context import context

##################
# Hypothesis optimization on a single device
# See Appendix B > Inference: overview > Hypothesis optimization
# This code loads config, observation, and hypothesis, then runs inference
##################


def solo_optimization(hypothesis, sound_name, sound_group, expt_name, hypothesis_name, config_name, seed=None, cuda_idx=0, round_type=None):
    """Infer variational posterior for a pre-defined hypothesis"""

    manual_seed(seed)
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
    device = inference.io.get_device(cuda_idx)

    observation, audio_sr = sf.read(os.path.join(
        os.environ["sound_dir"], sound_group, sound_name + ".wav"
        ))
    with context(experiment_config, batch_size=experiment_config["optimization"]["batch_size"]):
        scene_init = Scene.hypothesize(
            hypothesis,
            experiment_config["hyperpriors"],
            experiment_config["likelihood"],
            observation, audio_sr
            )

    scene_opt, optimizers, schedulers, metrics, hypothesis_config, savepath = hypothesis_setup(
        sound_name, sound_group, expt_name, hypothesis_name, seed, experiment_config, scene_init, device
        )
    with context(hypothesis_config, batch_size=hypothesis_config["optimization"]["batch_size"]):
        elbo_is = inference.optimize.basic_loop_from_scene(
            scene_opt, optimizers, schedulers, metrics,
            hypothesis_config, savepath=savepath, round_type=round_type
            )

    return scene_opt, elbo_is


def hypothesis_setup(sound_name, sound_group, expt_name, hypothesis_name=None, seed=None, experiment_config=None, scene_init=None, device=None, hierarchical=False):
    """Create (or reload) components necessary for optimizing a hypothesis"""
    savepath = inference.io.get_directory(
        sound_name, sound_group, expt_name, hypothesis_name=hypothesis_name, seed=seed
        )
    hypothesis_config = inference.io.get_config(
        savepath, experiment_config=experiment_config
        )
    hypothesis, checkpoint = inference.io.restore_hypothesis(
        savepath, scene_structure=scene_init
        )
    if device is not None:
        hypothesis.to(device)
    optimizers, schedulers = inference.optimizers.create(
        hypothesis, hypothesis_config["optimization"], checkpoint=checkpoint
        )
    if hierarchical:
        metrics = inference.metrics.SourcePriorLogger(checkpoint=checkpoint)
    else:
        metrics = inference.metrics.BasicLogger(
            checkpoint=checkpoint,
            metadata={'hypothesis_name': hypothesis_name, 'seed': seed}
            )
    hypothesis.train()
    print(hypothesis)
    return hypothesis, optimizers, schedulers, metrics, hypothesis_config, savepath