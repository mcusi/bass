import os
import sys
import yaml
import numpy as np
from glob import glob
from copy import deepcopy

import torch
from detectron2.engine import DefaultPredictor

from inference.amortized.configuration import get_base_cfg
from inference.amortized.dataloaders import AudioDataset
#Needs to be imported to add SoftOutputPanopticFPN to the "META_ARCH" registry
from inference.amortized import alt_mask_ops
from inference.amortized import detectron_to_latents, iou_util

###############
# Implements amortized inference as specified in the paper
# See Appendix B > Event proposals via amortized inference – segmentation neural network > Test procedure
###############


def load_network(detectron_expt, thresholds, base_cfg, checkpoint=None):
    """ Loads network to use for amortized inference """

    # Load weights
    cfg = deepcopy(base_cfg)
    list_of_weights = glob(os.path.join(os.environ["segmentation_dir"], detectron_expt, "*.pth"))
    if checkpoint is None:
        print(max(list_of_weights, key=os.path.getctime))
        cfg.MODEL.WEIGHTS = max(list_of_weights, key=os.path.getctime)
    else:
        ckpt_name = [k for k in list_of_weights if checkpoint in k][0]
        print("Using checkpoint:", ckpt_name)
        cfg.MODEL.WEIGHTS = [k for k in list_of_weights if checkpoint in k][0]
        print(cfg.MODEL.WEIGHTS)
    if "score" in thresholds.keys():
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresholds["score"]
    elif "scores" in thresholds.keys():
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(thresholds["scores"])
    else:
        raise Exception("No confidence score thresholds.")
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = thresholds["nms"]
    cfg.MODEL.META_ARCHITECTURE = "SoftOutputGeneralizedRCNN"

    if not torch.cuda.is_available():
        print("No CUDA, running on CPU.")
        cfg.MODEL.DEVICE = 'cpu'

    # Get prediction for each sound
    predictor = DefaultPredictor(cfg)

    return predictor, cfg


def default_thresholds():
    """ Default thresholds used for all sounds in the paper """

    thresholds_1 = {
        "iou": 0.5,
        "binary": 0.5,
        "nms": 0.5,
        "scores": [0.1, 0.1, 0.7],
        "iou_mask": "ibm",
        "min_proposals": 10
    }

    thresholds_2 = {
        "iou": 0.5,
        "binary": 0.5,
        "nms": 0.9,
        "scores": [0.1, 0.1, 0.7],
        "iou_mask": "f0",
        "min_proposals": 0
    }

    threshold_sets = [thresholds_1, thresholds_2]

    return threshold_sets


def get_predictions(d, predictor, thresholds):
    """ Run predictor on sounds and apply IoU thresholds to return Detectron2 output """

    # Run the neural network
    outputs = predictor(d["image"])
    # Format network outputs
    predictions = {
        "scores": outputs["instances"].scores.cpu(),
        "classes": outputs['instances'].pred_classes.cpu(),
        "boxes": outputs['instances'].pred_boxes.tensor.cpu(),
        "f0": getattr(outputs["instances"], "pred_f0", None),
        "amplitude": getattr(outputs["instances"], "pred_amplitude", None),
        "spa": getattr(outputs["instances"], "pred_spa", None),
        "ibm": outputs["instances"].pred_ibm.cpu()
    }

    # Calculate IOU matrix
    if thresholds["iou_mask"] == "ibm":
        # Use IBM for all IoUs
        iou_matrix = iou_util.create_iou_matrix(predictions["ibm"])
    elif thresholds["iou_mask"] == "f0":
        iou_matrix = iou_util.create_f0pred_iou_matrix(predictions)

    # Figure out which proposals should be kept based on score and IoU
    if "score" in thresholds.keys():
        # Same score threshold for all sound-types
        idx_to_keep = iou_util.iou_filter(iou_matrix, thresholds["iou"])
    elif "scores" in thresholds.keys():
        # Sound-type specific score threshold
        idx_to_keep = iou_util.iou_filter_by_class(
            iou_matrix,
            predictions["scores"],
            predictions["classes"],
            thresholds["scores"],
            thresholds["iou"]
            )

    predictions["iou_matrix"] = iou_matrix
    predictions["idx_to_keep"] = idx_to_keep

    return predictions


def combine_predictions(preds_from_predictors, threshold_sets):
    """ Combining the predictions across >1 predictor. """

    # If there is only one predictor, return those predictions
    if len(preds_from_predictors) == 1:
        return preds_from_predictors[0], preds_from_predictors[0]["iou_matrix"], preds_from_predictors[0]["idx_to_keep"]

    # Combine predictions from multiple predictors into single Tensors
    combined_predictions = {"predictor_idx": np.array([])}
    for predictor_idx, preds_from_one_predictor in enumerate(preds_from_predictors):
        idx_to_keep = preds_from_one_predictor["idx_to_keep"]
        # Keep track of which predictor these predictions came from
        combined_predictions["predictor_idx"] = np.concatenate((
            combined_predictions["predictor_idx"],
            np.full(len(idx_to_keep), predictor_idx)
            ))
        # Concatenate predictions
        for k, v in preds_from_one_predictor.items():
            if "iou_matrix" in k or "idx_to_keep" in k or "predictor_idx" in k:
                continue
            if k in combined_predictions:
                combined_predictions[k] = torch.cat((
                    combined_predictions[k], v[idx_to_keep, ...]
                    ))
            else:
                combined_predictions[k] = v[idx_to_keep, ...]

    # Get the IoU as specified by the later threshold set
    iou_matrix_ibm = iou_util.create_iou_matrix(combined_predictions["ibm"])
    iou_matrix_f0 = iou_util.create_f0pred_iou_matrix(combined_predictions)
    iou_matrix_combined = np.zeros(iou_matrix_ibm.shape)
    for i, predictor_idx_i in enumerate(combined_predictions["predictor_idx"]):
        for j, predictor_idx_j in enumerate(combined_predictions["predictor_idx"]):
            thresholds = threshold_sets[int(
                max([predictor_idx_i, predictor_idx_j])
                )]
            iou_matrix_combined[i, j] = iou_matrix_ibm[i, j] if thresholds["iou_mask"] == "ibm" else iou_matrix_f0[i, j]

    # Exclude duplicates
    combined_idx_to_keep = iou_util.iou_filter(
        iou_matrix_combined, max([thresholds["iou"] for thresholds in threshold_sets])
        )
    sorted_idx = np.argsort(combined_predictions["scores"].numpy())[::-1]
    sorted_idx = [si for si in sorted_idx if si in combined_idx_to_keep]

    return combined_predictions, iou_matrix_combined, sorted_idx


def proposals_from_predictions(d, predictions, threshold_sets, dataset, dataset_config, detectron_config, summary_fn, checkpoint=None):
    """ Convert detectron2 predictions into latent variables, for all events in a sound """

    # Get rid of overlapping predictions
    predictions, iou_matrix, idx_to_keep = combine_predictions(predictions, threshold_sets)
    # Get rid of silent predictions
    idx_to_keep = [idx for idx in idx_to_keep if not detectron_to_latents.silent_box(d["inputs"], predictions, idx)]

    # Format outputs to save
    d["outputs"] = {
        "idx_to_keep": idx_to_keep,
        "scores": predictions["scores"][idx_to_keep],
        "iou_matrix": iou_matrix[idx_to_keep, :][:, idx_to_keep],
        "boxes":  predictions["boxes"][idx_to_keep, :],  # 0: onsets, 2: offsets
        "classes": predictions["classes"][idx_to_keep],
        "thresholds": threshold_sets
    }
    for k in ["f0", "amplitude", "spa", "ibm"]:
        # Reshape masks so they are (n_events, t, f)
        d["outputs"][k] = predictions[k][idx_to_keep, ...].permute(0, 2, 1)

    # Obtain latent variables from detectron2 predictions, to include in outputs
    d["outputs"]["latents"] = []
    for kept_proposals_idx, detectron_output_idx in enumerate(idx_to_keep):   
        proposal = detectron_to_latents.make_event_proposal(detectron_output_idx, predictions, dataset.f_coch, dataset_config, detectron_config)
        proposal["rank"] = kept_proposals_idx
        d["outputs"]["latents"].append(proposal)

    return d


def amortized_inference(dataset_name, expt_name, sound_group):
    """ Obtain and save event proposals for sounds

        Inputs
        ------
        dataset_name (str): dataset used to train neural network
        expt_name (str): training run of neural network
        sound_group (str): folder name with sounds to do inference for
    """

    # Get folder to save results
    detectron_expt = "_".join([dataset_name, expt_name])
    summary_folder = os.path.join(os.environ["segmentation_dir"], detectron_expt, sound_group, "")
    os.makedirs(summary_folder, exist_ok=True)

    # Load config
    cfg = get_base_cfg()
    config_files = glob(os.environ["segmentation_dir"] + detectron_expt + "/*.yaml")
    expt_config_file = [f for f in config_files if "_dataset.yaml" not in f][0]
    dataset_config_file = [f for f in config_files if "_dataset.yaml" in f][0]
    cfg.merge_from_file(expt_config_file)
    with open(dataset_config_file, "r") as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)
    threshold_sets = default_thresholds()

    # Load dataset
    dataset = AudioDataset(dataset_name, sound_group, extra_path="", im_type="IBM", dataset_config=dataset_config, cfg=cfg)

    # Load networks - predictors are the same trained network, but with different thresholds
    predictors = []
    cfgs = []
    for thresholds in threshold_sets:
        _predictor, _cfg = load_network(detectron_expt, thresholds, cfg)
        predictors.append(_predictor)
        cfgs.append(_cfg)

    # Obtain predictions for each sound in AudioDataset
    for d in iter(dataset):
        print(d["sound_name"])
        summary_fn = os.path.join(summary_folder, d["sound_name"])
        predictions = []
        for predictor, cfg, thresholds in zip(predictors, cfgs, threshold_sets):
            # Get network outputs and filter based on nms, score, IoU, etc. specified in thresholds
            predictions.append(get_predictions(d, predictor, thresholds, dataset_config, cfg))
            n_proposals = sum([len(p["idx_to_keep"]) for p in predictions])
            if n_proposals >= thresholds["min_proposals"]:
                # All first network predictions included
                # Only supplemented with next network if there aren't many proposals, etc.
                print(f"Done, with {n_proposals} predictions.")
                break
            else:
                print(f"Only {n_proposals} proposals. Continuing to next set of thresholds!")
        d = proposals_from_predictions(d, predictions, threshold_sets, dataset, dataset_config, cfg, summary_fn)
        torch.save(d, summary_fn + ".tar")


if __name__ == "__main__":
    amortized_inference(
        "replicate_newnoise_uniform_v2",
        "rnu2_cgm_defaultrun",
        str(sys.argv[1])
    )
