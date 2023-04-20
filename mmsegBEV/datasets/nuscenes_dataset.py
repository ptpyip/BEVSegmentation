import copy
import random

import numpy as np
import torch

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from mmcv.parallel import DataContainer

from .builder import DATASETS
from .custom import CustomBEVDataset


@DATASETS.register_module()
class NuScenesDataset(CustomBEVDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames_queue = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
    # def pre_pipeline(self, results):
    #    # maybe for future use
    #     """Initialization before data preparation.

    #     Args:
    #         results (dict): Dict before data preprocessing.

    #             - img_fields (list): Image fields.
    #             - bbox3d_fields (list): 3D bounding boxes fields.
    #             - pts_mask_fields (list): Mask fields of points.
    #             - pts_seg_fields (list): Mask fields of point segments.
    #             - bbox_fields (list): Fields of bounding boxes.
    #             - mask_fields (list): Fields of masks.
    #             - seg_fields (list): Segment fields.
    #             - box_type_3d (str): 3D box type.
    #             - box_mode_3d (str): 3D box mode.
    #     """
    #     results["img_fields"] = []
    #     results["bbox3d_fields"] = []
    #     results["pts_mask_fields"] = []
    #     results["pts_seg_fields"] = []
    #     results["bbox_fields"] = []
    #     results["mask_fields"] = []
    #     results["seg_fields"] = []
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            data = self.get_data_info(i)
            if data is None:
                return None
            self.pre_pipeline(data)
            example = self.pipeline(data)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        data_sample = []
        meta_maps = []
        prev_scene_token, prev_pos, prev_angle = None
        
        frames = [frame['img'].data for frame in queue]
        for i, frame in enumerate(queue):
            meta_map = frame['img_metas'].data
            meta_map['prev_bev_exists'] = (meta_map['scene_token'] == prev_scene_token)
            
            if meta_map['prev_bev_exists']:
                tmp_pos = copy.deepcopy(meta_map['can_bus'][:3])
                tmp_angle = copy.deepcopy(meta_map['can_bus'][-1])
                
                meta_map['can_bus'][:3] -= prev_pos
                meta_map['can_bus'][-1] -= prev_angle
                
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
                
            else:
                prev_scene_token = meta_map['scene_token']      # for next prev_bev_exists
                
                prev_pos = copy.deepcopy(meta_map['can_bus'][:3])
                prev_angle = copy.deepcopy(meta_map['can_bus'][-1])
                
                meta_map['can_bus'][:3] = 0
                meta_map['can_bus'][-1] = 0
                
            meta_maps[i] = meta_map
            
        data_sample['img'] = DataContainer(torch.stack(frames), cpu_only=False, stack=True)
        data_sample['img_metas'] = DataContainer(meta_maps, cpu_only=True)
        queue = queue[-1]
        return data_sample

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        data = dict(
            sample_idx=info['token'],
            location=info["location"],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            data.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            data['ann_info'] = annos

        rotation = Quaternion(data['ego2global_rotation'])
        translation = data['ego2global_translation']
        can_bus = data['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return data

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate(self,results):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics
