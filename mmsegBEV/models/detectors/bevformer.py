# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
# from mmsegBEV.core import bbox3d2result
from mmsegBEV.models.detectors.mvx_two_stage import MVXTwoStageDetector
import time
import copy
import numpy as np

from mmsegBEV.models import DETECTORS
from ..builder import build_head

from ..bricks import run_time
from ..grid_mask import GridMask

from torch.nn.functional import interpolate

@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                #  pts_voxel_layer=None,
                #  pts_voxel_encoder=None,
                #  pts_middle_encoder=None,
                #  pts_fusion_layer=None,
                 img_backbone=None,
                #  pts_backbone=None,
                 img_neck=None,
                #  pts_neck=None,
                 seg_head=None,
                #  img_roi_head=None,
                #  img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,self).__init__(
                            #  pts_voxel_layer, pts_voxel_encoder,
                            #  pts_middle_encoder, pts_fusion_layer,
                             img_backbone, 
                            #  pts_backbone, 
                             img_neck, 
                            #  pts_neck,
                             seg_head, 
                            #  img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained
                            )
        
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        if seg_head:
            self.seg_head = build_head(seg_head)
        if not seg_head:
            # print("no seg head")
            assert True

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {'prev_bev': None, 'scene_token': None,'prev_pos': 0,'prev_angle': 0,}


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_test = interpolate(img, size=(256, 256))
            img_feats = self.img_backbone(img_test)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
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

    def forward_seg_train(self, 
                          img_feats, 
                          img_metas, 
                          prev_bev,
                          gt_masks_bev 
                          ):
                          
        # print("Forward seg train!")
        loss = self.seg_head(img_feats, img_metas, prev_bev, gt_masks_bev)
        return loss

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img=None,
                      img_metas=None, 
                      gt_masks_bev=None):
        """Forward training function.
        Args:
        Returns:
            dict: Losses of different branches.
        """
        # print("****Forward train from /data/ddoo/projects/bevseg/BEVSegmentation/mmsegBEV/models/detectors/bevformer.py****")
        # print(gt_masks_bev.shape)
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            # print("First time training! or prev_bev not exists")
            prev_bev = None
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        outputs = {}
        losses = self.forward_seg_train(img_feats, img_metas, prev_bev, gt_masks_bev)

        for name, val in losses.items():
            outputs[f"loss/{name}"] = val
        
        # print("****losses from /data/ddoo/projects/bevseg/BEVSegmentation/mmsegBEV/models/detectors/bevformer.py****")
        # print(outputs)
        
        return outputs

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results

    def simple_test_seg(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        bev_feat, outs = self.seg_head(x, img_metas, prev_bev=prev_bev)

        # bbox_list = self.pts_bbox_head.get_bboxes(
        #     outs, img_metas, rescale=rescale)
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ]
        return bev_feat, outs

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        mask_bev_list = []
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # mask_bev_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, out = self.simple_test_seg(
            img_feats, img_metas, prev_bev, rescale=rescale)
        
        mask_bev_list = [{
                'masks_bev': out[i]
        } for i in range(len(img_metas))]
        
        return new_prev_bev, mask_bev_list

    # def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
    #     """Test function without augmentaiton."""
    #     img_feats = self.extract_feat(img=img, img_metas=img_metas)

    #     bbox_list = [dict() for i in range(len(img_metas))]
    #     new_prev_bev, bbox_pts = self.simple_test_pts(
    #         img_feats, img_metas, prev_bev, rescale=rescale)
    #     for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #         result_dict['pts_bbox'] = pts_bbox
    #     return new_prev_bev, bbox_list