
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from mmsegBEV.models import TRANSFORMER

@TRANSFORMER.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    ...
