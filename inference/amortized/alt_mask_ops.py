import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY, ROI_HEADS_REGISTRY
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers.mask_ops import _do_paste_mask, BYTES_PER_FLOAT, GPU_MEM_LIMIT
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals

##################
# Custom detectron2 classes for soft masks
# in order to be able to apply segmentation
# framework to audio.
#
# See Appendix B
# B.2.1. Event proposals via amortized inference – segmentation neural network
##################


def my_paste_masks_in_image(masks, boxes, image_shape, tall):
    """ Based on `detectron2.layers.mask_ops.paste_masks_in_image` """

    img_h, img_w = image_shape

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    if tall:
        boxes = make_tall_boxes(Boxes(boxes), img_h).tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one
        # with skip_empty=True, so that it performs minimal
        # number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks,
        # but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds],
            img_h, img_w,
            skip_empty=(device.type == "cpu")
        )

        img_masks[(inds,) + spatial_inds] = masks_chunk

    return img_masks


def get_loss_function_for_mask_type(mask_type: str):
    if mask_type in ('bool', 'binary'):
        return lambda pred, gt: F.binary_cross_entropy_with_logits(
            pred, gt, reduction="mean"
            )
    elif mask_type in ('real', 'soft'):
        return lambda pred, gt: F.mse_loss(pred, gt, reduction="mean")
    elif mask_type == 'positive':
        return lambda pred, gt: F.mse_loss(pred, gt, reduction="mean")
    else:
        raise NotImplementedError()


def get_transform_function_for_mask_type(mask_type: str):
    if mask_type in ('bool', 'binary'):
        return lambda pred: F.sigmoid(pred)
    elif mask_type in ('real', 'soft'):
        return lambda pred: pred
    elif mask_type == 'positive':
        return lambda pred: pred.clip(min=0)
    else:
        raise NotImplementedError()


@torch.jit.unused
def my_mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], mask_types: Dict[str, str], tall:bool):
    """ Based on `detectron2.modeling.roi_heads.mask_head.mask_rcnn_loss`
        Added: mask types
    """

    N, n_output_channels, mask_side_len, _ = pred_mask_logits.shape
    num_classes = n_output_channels // len(mask_types)
    cls_agnostic_mask = num_classes == 1
    pred_mask_logits = pred_mask_logits.view(
        N, num_classes, len(mask_types), mask_side_len, mask_side_len
        )

    gt_classes = []
    gt_masks = {k: [] for k in mask_types}
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        for k in mask_types:
            boxes = instances_per_image.proposal_boxes
            if tall:
                height = instances_per_image._image_size[0]
                boxes = make_tall_boxes(boxes, height)
            gt_masks_per_image = getattr(
                instances_per_image, f"gt_{k}"
                ).crop_and_resize(
                    boxes.tensor, mask_side_len
                ).to(device=pred_mask_logits.device) 
            # A tensor of shape (N, M, M)
            # N=#instances in the image
            # M=mask_side_len
            gt_masks[k].append(gt_masks_per_image)

    for k in mask_types:
        gt_masks[k] = cat(gt_masks[k], dim=0).to(torch.float32)

    indices = torch.arange(N)
    gt_classes = cat(gt_classes, dim=0)
    pred_mask_logits = pred_mask_logits[indices, gt_classes]
    pred_mask_logits = {
        k: pred_mask_logits[:, i] for i, k in enumerate(mask_types)
        }

    mask_losses = {
        f"loss_{k}": get_loss_function_for_mask_type(v)(
            pred_mask_logits[k], gt_masks[k]
            )
        for k, v in mask_types.items()
    }

    return mask_losses


def my_mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances], mask_types:Dict[str, str]):
    """ Based on `detectron2.modeling.roi_heads.mask_head.mask_rcnn_inference` """
    N, n_output_channels, mask_side_len, _ = pred_mask_logits.shape
    num_classes = n_output_channels // len(mask_types)
    # cls_agnostic_mask = num_classes == 1
    pred_mask_logits = pred_mask_logits.view(
        N, num_classes, len(mask_types), mask_side_len, mask_side_len
        )

    class_pred = cat([i.pred_classes for i in pred_instances])
    indices = torch.arange(N, device=class_pred.device)
    pred_mask_logits = pred_mask_logits[indices, class_pred]
    # Keep a singleton dimension
    # (though do not know why this is in the original code?)
    pred_mask_logits = {
        k: pred_mask_logits[:, i:i+1] for i, k in enumerate(mask_types)
        }

    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_masks = {}
    for k, v in mask_types.items():
        f = get_transform_function_for_mask_type(v)
        pred_masks[k] = f(pred_mask_logits[k]).split(
            num_boxes_per_image, dim=0
            )

    for k in mask_types:
        for prob, instances in zip(pred_masks[k], pred_instances):
            setattr(instances, f"pred_{k}", prob)  # (1, Hmask, Wmask)


@ROI_HEADS_REGISTRY.register()
class MyStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, *,
            im_height,
            tallmask_pooler_resolution,
            tallmask_in_features: Optional[List[str]] = None,
            tallmask_pooler: Optional[ROIPooler] = None,
            tallmask_head: Optional[nn.Module] = None,
            **kwargs
            ):
        super().__init__(**kwargs)

        self.im_height = im_height
        self.tallmask_pooler_resolution = tallmask_pooler_resolution
        self.tallmask_on = tallmask_in_features is not None
        if self.tallmask_on:
            self.tallmask_in_features = tallmask_in_features
            self.tallmask_pooler = tallmask_pooler
            self.tallmask_head = tallmask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

        assert cfg.BASA.IM_HEIGHT > 0
        ret['im_height'] = cfg.BASA.IM_HEIGHT
        ret['tallmask_pooler_resolution'] = cfg.BASA.ROI_TALLMASK_HEAD.POOLER_RESOLUTION

        if len(cfg.BASA.TALLMASK_TYPES) > 0:
            cfg_tallmask = deepcopy(cfg)
            cfg_tallmask.MODEL.ROI_MASK_HEAD = cfg.BASA.ROI_TALLMASK_HEAD
            cfg_tallmask.MODEL.ROI_MASK_HEAD.TALL = True
            ret_tallmask = cls._init_mask_head(cfg_tallmask, input_shape)
            ret['tallmask_in_features'] = ret_tallmask['mask_in_features']
            ret['tallmask_pooler'] = ret_tallmask['mask_pooler']
            ret['tallmask_head'] = ret_tallmask['mask_head']
        return ret

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """ Forward logic of the mask prediction branch. """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(
                instances, self.num_classes
                )

        if self.mask_pooler is not None:
            mask_features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(mask_features, boxes)
        else:
            mask_features = {f: features[f] for f in self.mask_in_features}
        mask_output = self.mask_head(mask_features, instances)

        if self.tallmask_on:
            if self.tallmask_pooler is not None:
                tallmask_features = [
                    features[f] for f in self.tallmask_in_features
                    ]
                boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
                tall_boxes = [
                    make_tall_boxes(b, self.im_height) for b in boxes
                    ]
                tallmask_features = self.mask_pooler(
                    tallmask_features, tall_boxes
                    )
            else:
                tallmask_features = {
                    f: features[f] for f in self.tallmask_in_features
                    }

            H = self.im_height
            s = self.tallmask_pooler_resolution
            y0 = torch.cat([b.tensor[:, 1] for b in boxes])/H
            y1 = torch.cat([b.tensor[:, 3] for b in boxes])/H
            _y0 = - y0 / (y1-y0)
            _y1 = (1 - y0) / (y1-y0)
            N = len(y0)
            lin = torch.linspace(1/(2*s), 1-1/(2*s), s, device=y0.device)

            x = lin[None, :, None]
            y = _y0[:, None, None] + lin[None, None, :]*(_y1-_y0)[:, None, None]
            grid = torch.stack([
                -1 + 2*x.expand([N, s, s]),
                -1 + 2*y.expand([N, s, s]),
                ], 3)

            t = torch.ones(N, 1, s, s, device=y0.device)
            boxmask = F.grid_sample(t, grid).transpose(2, 3)
            tallmask_output = self.tallmask_head(
                boxmask*tallmask_features, instances
                )
            if self.training:
                mask_output = {**mask_output, **tallmask_output}

        return mask_output


def make_tall_boxes(boxes, height):
    box_tensor = boxes.tensor.clone()
    box_tensor[:, 1] = 0
    box_tensor[:, 3] = height
    return Boxes(box_tensor)


@ROI_MASK_HEAD_REGISTRY.register()
class MyMaskRCNNConvUpsampleHead(MaskRCNNConvUpsampleHead):
    """
    Based on `detectron2.modeling.roi_heads.mask_head.BaseMaskRCNNHead`
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", mask_types, tall, **kwargs):
        # Increasing the number of classes is a hack to increase the number of output channels
        super().__init__(
            input_shape, num_classes=num_classes*len(mask_types),
            conv_dims=conv_dims, conv_norm=conv_norm, **kwargs
            )
        self.mask_types = mask_types  # dict: name -> loss
        self.tall = tall

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        tall = getattr(cfg.MODEL.ROI_MASK_HEAD, "TALL", False)
        ret["tall"] = tall
        ret["mask_types"] = dict(
            cfg.BASA.TALLMASK_TYPES if tall else cfg.BASA.MASK_TYPES
            )
        return ret

    def forward(self, x, instances: List[Instances]):
        x = self.layers(x)
        if self.training:
            assert not torch.jit.is_scripting()
            return my_mask_rcnn_loss(
                x, instances, mask_types=self.mask_types, tall=self.tall
                )
        else:
            my_mask_rcnn_inference(x, instances, mask_types=self.mask_types)
            return instances



@META_ARCH_REGISTRY.register()
class SoftOutputGeneralizedRCNN(GeneralizedRCNN):
    """ Based on `detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN`"""

    def forward(self, batched_inputs):
        if self.training:
            losses = super().forward(batched_inputs)
            if not all(np.isfinite(v.item()) for v in losses.values()):
                print(
                    "-- Error: Loss became infinite or NaN --\n",
                    f"loss_dict = {losses}\n",
                    "Batch:\n" + "\n".join(
                        "  sound_group=" + x['sound_group'] + ", sound_name=" + x['sound_name']
                        for x in batched_inputs
                    )
                )
            elif sum(v.item() for v in losses.values()) > 10:
                print(
                    "-- Warning: BIG LOSS --\n",
                    f"loss_dict = {losses}\n",
                    "Batch:\n" + "\n".join(
                        "  sound_group=" + x['sound_group'] + ", sound_name=" + x['sound_name']
                        for x in batched_inputs
                    )
                )
            return losses
        else:
            return super().forward(batched_inputs)

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
                )

        if do_postprocess:
            # EDITED TO ADD SOFT MASKS
            # return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

            (instances, batched_inputs, image_sizes) = (
                results, batched_inputs, images.image_sizes
                )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)

                for k in self.roi_heads.mask_head.mask_types:
                    setattr(r, f"pred_{k}", 
                        retry_if_cuda_oom(my_paste_masks_in_image)(
                            getattr(r, f"pred_{k}")[:, 0, :, :],  # N, 1, M, M
                            r.pred_boxes,
                            r.image_size,
                            tall=False
                            # threshold=mask_threshold,
                        )
                    )

                if hasattr(self.roi_heads, "tallmask_head"):
                    for k in self.roi_heads.tallmask_head.mask_types:
                        setattr(r, f"pred_{k}",
                                retry_if_cuda_oom(my_paste_masks_in_image)(
                                    getattr(r, f"pred_{k}")[:, 0, :, :],  # N, 1, M, M
                                    r.pred_boxes,
                                    r.image_size,
                                    tall=True
                                )
                                )

                processed_results.append({"instances": r})
            return processed_results
        else:
            return results
