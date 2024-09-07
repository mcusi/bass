from detectron2.utils.logger import setup_logger
setup_logger()

import os
import sys
from shutil import copyfile
from glob import glob

import torch
from detectron2.engine import launch
import sys

from inference.amortized.dataloaders import BassTrainer
from util.sample import manual_seed
from inference.amortized.configuration import get_base_cfg
# Needs to be imported to add SoftOutputPanopticFPN to the "META_ARCH" registry
from inference.amortized import alt_mask_ops

manual_seed(0)

# Relevant directories
code_dir = os.path.join(os.environ["home_dir"], "inference", "amortized", "")
results_dir = os.environ["segmentation_dir"]
dream_dir = os.environ["dream_dir"]


def train(dataset_name, experiment_name, config_name):
    """ Train detectron on sounds created by dataset.py """

    print("Setting up config...", flush=True)
    cfg = get_base_cfg()
    cfg.OUTPUT_DIR = os.path.join(
        results_dir, f"{dataset_name}_{experiment_name}", ""
        )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.merge_from_file(os.path.join(code_dir, config_name + ".yaml"))
    copyfile(
        os.path.join(code_dir, config_name + ".yaml"),
        os.path.join(cfg.OUTPUT_DIR, config_name + ".yaml")
        )
    data_creation_config_fn = os.path.join(
        dream_dir, dataset_name + "_0", "data-creation_config.yaml"
        )
    copyfile(
        data_creation_config_fn,
        os.path.join(cfg.OUTPUT_DIR, dataset_name + "_dataset.yaml")
        )

    # ~~~~
    print("Setting up optimizers", flush=True)
    print("Existing paths:", glob(cfg.OUTPUT_DIR + "*.pth"))
    n_ckpts = len(glob(cfg.OUTPUT_DIR + "*.pth"))
    resume = True if n_ckpts > 0 else False
    cfg.DATASETS.TRAIN = dataset_name
    cfg.DATASETS.TEST = dataset_name

    trainer = BassTrainer(cfg)
    # should load the latest checkpoint if r > 0
    trainer.resume_or_load(resume=resume)
    print("Let's train", flush=True)
    trainer.train()


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    experiment_name = sys.argv[2]
    config_name = sys.argv[3]
    print(f"Dataset directory: {dream_dir}, dataset: {dataset_name}, experiment_name: {experiment_name}", flush=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        train(dataset_name, experiment_name, config_name)
    else:
        launch(
            train,
            num_gpus,
            dist_url="auto",
            args=(dataset_name, experiment_name, config_name)
        )
