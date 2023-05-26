from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION

from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS)

from mmdet.models.utils.builder import TRANSFORMER

MODELS = Registry('models', parent=MMCV_MODELS)
# ATTENTION = Registry('attention', parent=MMCV_ATTENTION )

# BACKBONES = MODELS
# NECKS = MODELS
# HEADS = MODELS
# LOSSES = MODELS
SEGMENTORS = MODELS
# DETECTORS = MODELS

def build_backbone(cfg):
    return BACKBONES.build(cfg)

def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)

def build_transformer(cfg):
    return TRANSFORMER.build(cfg)

def build_loss(cfg):
    return LOSSES.build(cfg)

def build_segmentor(cfg):
    return SEGMENTORS.build(cfg)

def build_detector(cfg, train_cfg=None, test_cfg=None):
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
