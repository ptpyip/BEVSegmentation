from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import interpolate

from ..builder import HEADS, build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding

__all__ = ["BEVSegmentationHead"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    # targets = interpolate(targets, size=(50, 50))
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
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
        
        u, v = torch.meshgrid(coords, indexing="ij")
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
        transformer: None,
        positional_encoding: None,
        grid_transform: None
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.num_query = num_query
        
        self.loss = loss
        self.transformer = build_transformer(transformer)
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
        self.query_embedding = nn.Embedding(self.num_query, 256 * 2)

    def forward(
        self,
        x,
        img_metas,
        prev_bev,
        target=None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        bs, num_cam, _, _, _ = x[0].shape
        dtype = x[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        x = self.transformer.get_bev_features(mlvl_feats=x, 
                                              bev_queries=bev_queries,
                                              object_query_embed=object_query_embeds, 
                                              bev_h=self.bev_h, bev_w=self.bev_w,
                                              bev_pos=bev_pos,
                                              prev_bev=prev_bev,
                                              img_metas=img_metas)

        

        if target is None:
            return x

        print(f"x : {len(x)}")
        print(f"x[0] : {x[0].shape}")

        # (1, 200*200, 256)
        x = torch.reshape(x, (1, self.bev_h, self.bev_w, -1))  #(BS, h, w, dim) = (1, 200, 200, 256)
        x = torch.permute(x, (0, 3, 1, 2)) # (BS, dim, h, w) = (1, 256, 200, 200)

        print(f"x : {x.shape}")

        x = self.transform(x)
        x = self.classifier(x)

        print("From /data/ddoo/projects/bevseg/BEVSegmentation/mmsegBEV/models/heads/segm.py, line 173: ")
        print(x.shape)

        losses = dict()
        losses['loss'] = sigmoid_focal_loss(x, target)

        return losses
    """
        def forward(
            self,
            x: torch.Tensor,
            target: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, Dict[str, Any]]:
            if isinstance(x, (list, tuple)):
                x = x[0]

            x = self.transform(x)
            x = self.classifier(x)

            if self.training:
                losses = {}
                for index, name in enumerate(self.classes):
                    if self.loss == "xent":
                        loss = sigmoid_xent_loss(x[:, index], target[:, index])
                    elif self.loss == "focal":
                        loss = sigmoid_focal_loss(x[:, index], target[:, index])
                    else:
                        raise ValueError(f"unsupported loss: {self.loss}")
                    losses[f"{name}/{self.loss}"] = loss
                return losses
            else:
                return torch.sigmoid(x)
    """
