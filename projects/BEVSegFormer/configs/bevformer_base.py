_base_ = [
    'datasets/nuscenes.py',
    './__base__/default_runtime.py'
]

seed = 0
deterministic = False

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# image_size = [256, 704]
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 2 # each sequence contains `queue_length` frames.

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        init_cfg = dict(type='Pretrained', checkpoint='ckpts/resnet101-5d3b4d8f.pth'),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    seg_head=dict(
        type='BEVSegmentationHead',
        in_channels=_dim_,
        loss='focal',
        num_classes=6,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1
                        ),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            deformable_attention=dict( 
                                        type='MSDeformableAttention3D', 
                                        embed_dims=256, num_points=8, num_levels=4),
                            embed_dims=256
                        )
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                )
            ),
            # decoder=dict(
            #     type='DetectionTransformerDecoder',
            #     num_layers=6,
            #     return_intermediate=True,
            #     transformerlayers=dict(
            #         type='DetrTransformerDecoderLayer',
            #         attn_cfgs=[
            #             dict(
            #                 type='MultiheadAttention',
            #                 embed_dims=_dim_,
            #                 num_heads=8,
            #                 dropout=0.1),
            #              dict(
            #                 type='CustomMSDeformableAttention',
            #                 embed_dims=_dim_,
            #                 num_levels=1),
            #         ],

            #         feedforward_channels=_ffn_dim_,
            #         ffn_dropout=0.1,
            #         operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
            #                          'ffn', 'norm'))
            # )
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True
        ),
        grid_transform=dict(
            input_scope = [[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope = [[-50, 50, 0.5], [-50, 50, 0.5]]
        )
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=2.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        # loss_iou=dict(type='GIoULoss', loss_weight=0.0)
    ),
    # model training and testing settings
    train_cfg=dict(
        # pts=dict(
        #     grid_size=[512, 512, 1],
        #     voxel_size=voxel_size,
        #     point_cloud_range=point_cloud_range,
        #     out_size_factor=4,
        #     assigner=dict(
        #         type='HungarianAssigner3D',
        #         cls_cost=dict(type='FocalLossCost', weight=2.0),
        #         reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
        #         iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
        #         pc_range=point_cloud_range)
        # )
    )
)


optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline={{_base_.test_pipeline}})

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=10)
