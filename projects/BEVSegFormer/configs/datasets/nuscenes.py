dataset_type = "NuScenesDataset"
data_root = '/data/ddoo_share/BEVSegmentation/data/nuscenes/'
file_client_args = dict(backend='disk')

point_cloud_range = [-50, -50, -5, 50, 50, 3]

# For nuScenes we usually do 10-class detection
class_names = [
    'drivable_area', 'ped_crossing', 'walkway', 
    'stop_line', 'carpark_area', 'divider'
]

map_classes = ['drivable_area', 'ped_crossing', 
               'walkway', 'stop_line',
               'carpark_area', 'divider']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

# image_size = [256, 704]
bev_h_ = 100
bev_w_ = 100
queue_length = 2 # each sequence contains `queue_length` frames.

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

map_classes = ['drivable_area', 'ped_crossing', 
               'walkway', 'stop_line',
               'carpark_area', 'divider']

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
)


augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(
        prob=0.0,
        fixed_prob=True       
    )
)

train_pipeline = [
    dict(type='ImageAug'),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='GlobalRotScaleTrans', 
        resize_lim=(1.0, 1.0), 
        rot_lim=(0.0, 0.0), 
        trans_lim=0.0, 
        is_train=False
    ),
    dict(
        type='LoadBEVSegmentation', 
        dataset_root=data_root, 
        xbound=[-50.0, 50.0, 0.5], 
        ybound=[-50.0, 50.0, 0.5],
        classes=map_classes,
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CustomCollect3D', keys=['img', 'gt_masks_bev'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='GlobalRotScaleTrans', 
        resize_lim=(1.0, 1.0), 
        rot_lim=(0.0, 0.0), 
        trans_lim=0.0, 
        is_train=False
    ),
    dict(
        type='LoadBEVSegmentation', 
        dataset_root=data_root, 
        xbound=[-50.0, 50.0, 0.5], 
        ybound=[-50.0, 50.0, 0.5],
        classes=map_classes
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
                dict(type='DefaultFormatBundle',with_label=False),
                dict(type='CustomCollect3D', keys=['img'])]
        ),
]

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/ddoo_share/BEVSegmentation/data/nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        palette=MAP_PALETTE,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file='/data/ddoo_share/BEVSegmentation/data/nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,  
        bev_size=(bev_h_, bev_w_),
        classes=class_names, 
        modality=input_modality, 
        # samples_per_gpu=1
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/ddoo_share/BEVSegmentation/data/nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffle=True,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
