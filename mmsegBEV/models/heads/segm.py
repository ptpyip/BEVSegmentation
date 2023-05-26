from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_
from torch.nn.functional import interpolate

from ..builder import HEADS, build_transformer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.transformer import build_positional_encoding
from torchvision.transforms.functional import rotate

import os
from mmsegBEV.core.utils import visualize_map

import numpy as np

__all__ = ["BEVSegmentationHead"]

map_classes = ['drivable_area', 'ped_crossing', 
               'walkway', 'stop_line',
               'carpark_area', 'divider']


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)
    
    if loss < 0.5:
        torch.save({
            "inputs_shape": inputs.shape,
            "targets_shape": targets.shape,
            "inputs": inputs,
            "targets": targets,
        }, "work_dirs/bevformer_base/temp_out.pth")

    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    # alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()

    p = torch.sigmoid(inputs)
        
    eps = 1e-8
    ce_loss = F.binary_cross_entropy_with_logits(inputs + eps, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    
    if loss < 0.04:
        torch.save({
            "inputs_shape": inputs.shape,
            "targets_shape": targets.shape,
            "inputs": inputs,
            "targets": targets,
        }, "work_dirs/bevformer_base/temp_out.pth")
    
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))
            
        
        u, v = torch.meshgrid(coords)
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        loss: str,
        num_classes: int,
        bev_h : int,
        bev_w : int,
        num_query : int,
        rotate_center=[100, 100],
        use_shift = True,
        rotate_prev_bev = True,
        use_can_bus = True,
        use_cams_embeds = True,
        transformer = None,
        positional_encoding = None,
        grid_transform = None,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.num_query = num_query
        
        self.rotate_center = rotate_center
        self.use_shift = use_shift
        self.rotate_prev_bev = rotate_prev_bev
        self.use_can_bus = use_can_bus
        self.use_cams_embeds = use_cams_embeds
        self.embed_dims = 256
        self.num_cams = 6
        self.num_feature_levels = 4

        self.loss = loss

        self.encoder = build_transformer_layer_sequence(transformer.get('encoder'))
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transform = BEVGridTransform(**grid_transform)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self._init_layers()

    def _init_layers(self):
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, 256)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    def forward(
        self,
        x,
        img_metas,
        prev_bev,
        target=None,
        only_bev=False
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        bs, num_cam, _, _, _ = x[0].shape
        dtype = x[0].dtype
        
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_feat = self.get_bev_features(mlvl_feats=x, 
                                    bev_queries=bev_queries,
                                    bev_h=self.bev_h, bev_w=self.bev_w,
                                    bev_pos=bev_pos,
                                    prev_bev=prev_bev,
                                    img_metas=img_metas)

        if only_bev:
            return bev_feat
        
        x = bev_feat

        # (1, 100*100, 256)
        x = torch.reshape(x, (bs, self.bev_h, self.bev_w, -1))  #(BS, h, w, dim) = (1, 100, 100, 256)
        x = torch.permute(x, (0, 3, 1, 2)) # (BS, dim, h, w) = (1, 256, 200, 200)

        x = self.transform(x)
        x = self.classifier(x)
        
        if target is None:
            return bev_feat, torch.sigmoid(x)
        else:
            losses = dict()
            losses['loss'] = sigmoid_xent_loss(x, target)
            return losses
            
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor([each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        feat_flatten = torch.cat(feat_flatten, 2)
        
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed