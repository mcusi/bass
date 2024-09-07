from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import CfgNode as CN


def get_base_cfg():
    """ default config for training event proposan network """
    cfg = get_cfg()

    cfg.BASA = CN()
    cfg.BASA.IM_HEIGHT = -1
    cfg.BASA.IMS_PER_BATCH = 4
    cfg.BASA.TF_REPRESENTATION = "cgm"
    cfg.BASA.IM_CUTOFF_CEIL = -1
    cfg.BASA.USE_REL_THRESHOLD = True
    cfg.BASA.MASK_TYPES = []
    cfg.BASA.TALLMASK_TYPES = []
    cfg.BASA.BOX_HEIGHT_MASK = "ibm"
    cfg.BASA.USE_REAL_TIMEPOINTS = False
    cfg.BASA.USE_BACKGROUND_NOISE = False

    cfg.BASA.ROI_TALLMASK_HEAD = CN()
    cfg.BASA.ROI_TALLMASK_HEAD.CLS_AGNOSTIC_MASK = False
    cfg.BASA.ROI_TALLMASK_HEAD.CONV_DIM = 256
    cfg.BASA.ROI_TALLMASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.BASA.ROI_TALLMASK_HEAD.NORM = ""
    cfg.BASA.ROI_TALLMASK_HEAD.NUM_CONV = 4
    cfg.BASA.ROI_TALLMASK_HEAD.POOLER_RESOLUTION = 14
    cfg.BASA.ROI_TALLMASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.BASA.ROI_TALLMASK_HEAD.POOLER_TYPE = "ROIAlignV2"

    cfg.merge_from_file(model_zoo.get_config_file("./Base-RCNN-FPN.yaml"))
    return cfg
