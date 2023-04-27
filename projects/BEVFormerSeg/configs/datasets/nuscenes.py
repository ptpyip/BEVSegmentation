dataset_type = "NuScenesDataset"
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

point_cloud_range = [-50, -50, -5, 50, 50, 3]

# For nuScenes we usually do 10-class detection
class_names = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider'
]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

image_size = [256, 704]
_dim_ = 256
load_dim = use_dim = 5
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.


augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(
        prob=0.0,
        fixed_prob=True       
    )
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadBEVSegmentation', 
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(
    #     type='ImagAug', 
    #     final_dim=image_size,
    #     augment2d=augment2d,
    #     resize_lim=load_dim,
    #     bot_pct_lim=[0.0, 0.0],
    #     rand_flip=True,
    #     is_trian=True,
    #     is_used=False
    # ),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle', class_names=class_names),
    dict(type='CustomCollect', keys=['img', 'gt_masks_bev'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadBEVSegmentation', 
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        test_mode=False,
    ),
    val=dict(type=dataset_type,
             data_root=data_root,
            test_mode=False,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffle=True,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
