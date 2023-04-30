import torch
from torch import nn

from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding

from mmsegBEV.models import TRANSFORMER

@TRANSFORMER.register_module()
class BEVFormer(BaseModule):
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
                 encoder,
                 in_channels=256,
                 num_input_levels=4,
                 num_cams=6,
                 bev_h=200,
                 bev_w=200,
                 num_query=100,
                 num_classes=10,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 positional_encoding=None,
                 **kwargs):
        # super(BEVFormer, self).__init__(
        #     transformerlayers, num_layers)
        
        self.encoder = build_transformer_layer_sequence(encoder)
        
        self.positional_encoding = build_positional_encoding(
            positional_encoding if positional_encoding is not None else dict(
                type='LearnedPositionalEncoding',
                num_feats=in_channels,
                row_num_embed=bev_h,
                col_num_embed=bev_w
            )
        ) 
        
        self.num_input_level = num_input_levels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = self.encoder.embed_dims
        
        # flags 
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.use_cams_embeds = use_cams_embeds
        
        self.init_layers(num_cams, num_input_levels, can_bus_norm)
        
    def init_layers(self, num_cams, num_input_levels, can_bus_norm=True):
        """Initialize layers of the Detr3DTransformer."""
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims
        )
        
        self.query_embedding = nn.Embedding(
            self.num_query, self.embed_dims * 2
        )
            
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
      
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        # print(f"obtain_history_bev features: ")
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, img_metas=img_metas_list, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.seg_head(img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            # print(f"  prev_bev from history : {torch.any(torch.isnan(prev_bev))} ")
            return prev_bev      
    
    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, only_bev=False):
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        
        bev_embed = self.encoder()

