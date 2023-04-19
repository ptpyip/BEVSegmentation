from typing import List

import numpy as np

from dataclasses import dataclass

class AnnoInfo:
    gt_bboxes_3d: object            # LiDARInstance3DBoxes 3D ground truth bboxes
    gt_labels_3d: np.ndarray        # Labels of ground truths.
    gt_names: List[str]             # Class names of ground truths.     
    gt_masks_bev: np.ndarray        # (N, C, H, W)  
    
class PoseInfo:
    ''' From CAM bus '''
    
    translation: np.ndarray     # 0-2 | ego2global translation
    rotation: np.ndarray        # 3-6| ego2global rotation
    accel: float                # 7-9
    rotation_rate: float        # 10-12
    velocity: float             # 13-15
    yaw_angle: float            # 16
    patch_angle: float          # 17
    
class nuScencesImage:
    img: np.ndarray
    
    
class MetaInfo:
    sample_idx: str                 # info["token"]
    timestamp: float                # info["token"]
    location: str                   # info["location"]
    pts_filename: str               # Filename of point clouds; info["lidar_path"]
    sweeps_info: List[dict]         # info["sweeps"]
    
    img_filename: str               # info["token"]
    camera_intrinsics: np.ndarray      
    
    # projection matrixes
    ego2global: np.ndarray      
    lidar2ego: np.ndarray  
    lidar2camera: np.ndarray
    camera2ego: np.ndarray
    camera2lidar: np.ndarray
    
@dataclass
class CameraMetric:
    camera_intrinsics: np.ndarray
    camera2ego: np.ndarray
    lidar2image: np.ndarray
    cam2img: np.ndarray
    camera2ego: np.ndarray
       
class nuScencesData:
    ''''''
    
    ## data
    img: np.ndarray
    gt_mask_bev: np.ndarray
    
    ## meta
    PoseInfo: List
    aug_matrix: np.ndarray
    
    ## cam_metrics
    
    