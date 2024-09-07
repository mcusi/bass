import numpy as np
from typing import Any, List, Union

import torch
from detectron2.layers.roi_align import ROIAlign
from detectron2.structures.boxes import Boxes


class SoftMasks:
    """ Adapted from BitMasks."""

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, *args: Any, **kwargs: Any) -> "SoftMasks":
        return SoftMasks(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> "SoftMasks":
        if isinstance(item, int):
            return SoftMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on SoftMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return SoftMasks(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        return self.tensor.flatten(1).any(dim=1)

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )
        return output

    def get_bounding_boxes(self) -> Boxes:
        raise NotImplementedError()

    @staticmethod
    def cat(bitmasks_list: List["SoftMasks"]) -> "SoftMasks":
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, SoftMasks) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks
