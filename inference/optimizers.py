import re
from collections import OrderedDict

import torch
import torch.optim

import inference.io
import inference.metrics

#################
# Optimizer creation
# See Appendix B > Inference: hypothesis optimization and scene selection
#################


def create_param_groups(scene, config, verbose=False):
    """ Group parameters for setting learning rates in optimizer """
    param_groups = OrderedDict()

    if "learning_rates" not in config.keys():
        print("WARNING! Using default learning rates as coded in optimize.py.")
    LR = config.get("learning_rates", {})

    # --- Things that affect likelihood ---
    param_groups["f0 inducing values"] = {
        'pattern': r"f0\..*\._variational_mean",
        'lr': LR.get("f0_inducing_values", 5)
    }
    param_groups["Spectrum/amplitude inducing values"] = {
        'pattern': r"(spectrum|amplitude)\..*\._variational_mean",
        'lr': LR.get("spectrum_amplitude_inducing_values", 1)
    }
    param_groups["Inducing standard deviation"] = {
        'pattern': r"(spectrum|f0|amplitude)\..*\._variational_stddev",
        'lr': LR.get("inducing_standard_deviation", 1)
    }
    param_groups["Temporal inducing points"] = {
        'pattern': r"(f0|amplitude)\..*\._?inducing_points",
        'lr': LR.get("temporal_inducing_points", 0.1)
    }
    param_groups["Spectrum inducing points"] = {
        'pattern': r"spectrum\..*\._?inducing_points",
        'lr': LR.get("spectrum_inducing_points", 0.1)
    }
    param_groups["Timepoints mu"] = {
        'pattern': r"(onset|offset)\._mu",
        'lr': LR.get("timepoints_mu", 0.05)
    }
    param_groups["Timepoints sigma"] = {
        'pattern': r"(onset|offset)\._sigma",
        'lr': LR.get("timepoints_sigma", 1)
    }

    # --- Things that don't affect likelihood ---
    param_groups["GP lengthscale"] = {
        'pattern': r"covar_module\.hyperparams\.scale\.(mu|sigma)",
        'lr': LR.get("gp_lengthscale", 10)
    }
    param_groups["GP sigma"] = {
        'pattern': r"covar_module\.hyperparams\.sigma\.(mu|sigma)",
        'lr': LR.get("gp_sigma", 1)
    }
    param_groups["GP mu"] = {
        'pattern': r"mean_module\._mu_(mu|sigma)",
        'lr': LR.get("gp_mu", 1)
    }
    param_groups["GP sigma_within"] = {
        'pattern': r"covar_module\.hyperparams\.sigma_within\.(mu|sigma)",
        'lr': LR.get("gp_sigma_within", 1)
    }
    param_groups["GP sigma_residual"] = {
        'pattern': r"covar_module\.hyperparams\.sigma_residual\.(mu|sigma)",
        'lr': LR.get("gp_sigma_residual", 1)
    }
    param_groups["Sequence timings distribution"] = {
        'pattern': r"sequence\.(gap|duration)\..*",
        'lr': LR.get("sequence_timings_distribution", 1)
    }
    param_groups["GP hyperpriors"] = {
        'pattern': r"feature_hp",
        'lr': LR.get("gp_hyperpriors", 1)
    }
    param_groups["GP epsilon"] = {
        'pattern': r"feature._epsilon",
        'lr': LR.get("gp_epsilon", 1)
    }
    param_groups["Sequence hyperpriors"] = {
        'pattern': r"sequence_hp",
        "lr": LR.get("sequence_hyperpriors", 1)
    }

    for k in param_groups:
        param_groups[k]['params'] = []
        param_groups[k]['param_names'] = []
    for (n, p) in scene.named_parameters():
        for k, v in param_groups.items():
            if re.search(v['pattern'], n):
                v['params'].append(p)
                v['param_names'].append(n)
                break
        else:
            raise Exception(f"No match for parameter {n}")

    for k, v in param_groups.items():
        if len(v['params']) == 0:
            print(f"WARNING: No params match group: {k}")
        if verbose:
            print(f"Params: {k} (lr = {v['lr']})")
            for n in v['param_names']:
                print(f"   {n}")
    return param_groups


def create(scene, config, savepath=None, checkpoint=None, device=None):
    """ Create optimizer for a particular Scene and its config """

    if config["multiple_optimizers"]:
        param_groups = create_param_groups(scene, config, verbose=False) 

        optimizer_class = getattr(torch.optim, config["optimizer_class"])
        optimizer = optimizer_class([
            {'params': v['params'], 'lr':config["learning_rate"] * v['lr']}
            for v in param_groups.values()
        ])

        optimizers = [optimizer]
    else:
        optimizer_class = getattr(torch.optim, config["optimizer_class"])
        optimizer = optimizer_class(scene.parameters(), lr=config["learning_rate"]) 
        optimizers = [optimizer]

    if config["schedule"]:
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(ozr, 'min', patience=config["schedule_patience"]) for ozr in optimizers]
    else:
        schedulers = None

    if (savepath is not None) or (checkpoint is not None):
        inference.io.restore_optimizers(optimizers, schedulers, savepath=savepath, checkpoint=checkpoint, device=device)

    return optimizers, schedulers
