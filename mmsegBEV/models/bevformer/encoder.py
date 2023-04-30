import torch
from torch import nn

from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from mmsegBEV.models import TRANSFORMER

@TRANSFORMER.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    """Implements the BEVFormer Encoder.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_input_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    rotate_center=[100, 100]
    
    def __init__(self,
                 transformerlayers,
                 num_layers=4,
                 num_input_levels=4,
                 num_cams=6,
                 bev_h=200,
                 bev_w=200,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 **kwargs):
        super(BEVFormerEncoder, self).__init__(
            transformerlayers, num_layers)
        
        self.num_input_level = num_input_levels
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # flags 
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.use_cams_embeds = use_cams_embeds
        
        self.init_layers(num_cams, num_input_levels, can_bus_norm)
        
    def init_layers(self, num_cams, num_input_levels, can_bus_norm=True):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            num_input_levels, self.embed_dims
        ))
        
        self.cams_embeds = nn.Parameter(torch.Tensor(
            num_cams, self.embed_dims
        ))
        
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
