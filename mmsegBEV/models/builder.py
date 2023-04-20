from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS

def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
