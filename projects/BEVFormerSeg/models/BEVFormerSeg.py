
import time
import copy
import numpy as np
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmsegBEV.models import SEGMENTORS
from mmsegBEV.models import build_transformer, build_head
from mmsegBEV.models.segmentors import EncoderDecoder
# from bricks import run_time
from mmsegBEV.models.utils import GridMask

@SEGMENTORS.register_module()
class BEVFormerSeg(EncoderDecoder):
    '''
        BEVFormerSeg - The implementation of BEVFormer on BEV Segmentation tasks
    '''
    
    def __init__(
        self, 
        img_backbone, 
        decoder_head, 
        img_neck=None, 
        bev_encoder=None,
        auxiliary_head=None, 
        train_cfg=None, 
        test_cfg=None, 
        pretrained=None, 
        init_cfg=None,
        use_grid_mask=False,
        video_test_mode=False,
    ):
        super(BEVFormerSeg, self).__init__(
            auxiliary_head, train_cfg, test_cfg, pretrained, init_cfg,
            backbone=img_backbone, decoder_head=decoder_head, neck=img_neck
        )
        
        if bev_encoder != None:
            self.bev_encoder = build_transformer(bev_encoder)
            
        self.grid_mask = use_grid_mask
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
            
        
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        return
    
        
    def _init_decoder_head(self, decoder_head):
        """Initialize ``decoder_head`` in EncoderDecoder"""
        self.decoder_head = build_head(decoder_head)
    
    
    ### Image Encoder: Backbone + Neck
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        if img is None: return None
        batch_size = img.size(0)
        
        img = self.preprocess_img(img)
        
        img_feats = self.backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        
        if self.with_neck:
            img_feats = self.neck(img_feats)
        
        return self.reshape_img_feats(img_feats, batch_size, len_queue=len_queue)
        
    def preprocess_img(self, img):
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            
        if self.use_grid_mask:
            img = self.grid_mask(img)

        return img

    def reshape_img_feats(self, img_feats, batch_size, len_queue=None):
        """Reshape features of images."""
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(batch_size/len_queue), len_queue, int(BN / batch_size), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(batch_size, int(BN / batch_size), C, H, W))
                
        return img_feats_reshaped
    
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
        
    
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(
        self,
        img=None,
        img_metas=None,
        gt_masks_bev=None
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        
        # prev_bev = None
        # if img_metas[0]['prev_bev_exists']:
        #     prev_img_metas = copy.deepcopy(img_metas)
        #     prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        img_metas = [img_meta[len_queue-1] for img_meta in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
            
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bev_feats = self.bev_encoder(
            img_feats, img_metas, prev_bev
        )
        out = self.decoder_head(bev_feats)
        loss = self.decoder_head.loss(out, gt_masks_bev, img_metas)
        
        return loss
    
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

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            ...