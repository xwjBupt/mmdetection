train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=50,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0025000000000000005,
        begin=50,
        T_max=950,
        end=950,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=950, end=1000)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
auto_scale_lr = dict(enable=True, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=10, save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    git_info=
    'COMMIT TAG [\nyolox_l_8xb8-1000e-lr5e-2_stenosis_binary_F0/fold0\nCOMMIT BRANCH >>> stenosis <<< \nCOMMIT ID >>> e993b25fe6436c143f996f7d24de2d37b7ca5e2d <<<]\n'
)
vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(project='CEREBRAL_STENOSIS', name=None, group=None))
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='CEREBRAL_STENOSIS',
                name='fold0',
                group='yolox_l_8xb8-1000e-lr5e-2_stenosis_binary_F0'))
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/ai/mnt/code/mmdetection/work_dirs/yolox_l_8xb8-1000e-lr5e-2_stenosis_binary_F0/best_coco_bbox_mAP_epoch_300.pth'
resume = False
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=100))
img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (960, 960),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'Pad',
                        'pad_to_square': True,
                        'pad_val': {
                            'img': (114.0, 114.0, 114.0)
                        }
                    }], [{
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]
img_scale = (640, 640)
model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = '/ai/mnt/data/stenosis/selected/Binary/FOLD0/'
dataset_type = 'CocoStenosisBinaryDataset'
backend_args = None
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoStenosisBinaryDataset',
        data_root='/ai/mnt/data/stenosis/selected/Binary/FOLD0/',
        ann_file='annotations/train_binary.json',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Resize', scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='PackDetInputs')
    ])
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoStenosisBinaryDataset',
            data_root='/ai/mnt/data/stenosis/selected/Binary/FOLD0/',
            ann_file='annotations/train_binary.json',
            data_prefix=dict(img='train/'),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            backend_args=None),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoStenosisBinaryDataset',
        data_root='/ai/mnt/data/stenosis/selected/Binary/FOLD0/',
        ann_file='annotations/val_binary.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoStenosisBinaryDataset',
        data_root='/ai/mnt/data/stenosis/selected/Binary/FOLD0/',
        ann_file='annotations/val_binary.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/ai/mnt/data/stenosis/selected/Binary/FOLD0/annotations/val_binary.json',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/ai/mnt/data/stenosis/selected/Binary/FOLD0/annotations/val_binary.json',
    metric='bbox',
    backend_args=None)
max_epochs = 1000
num_last_epochs = 50
interval = 10
base_lr = 0.05
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
dataset_name = 'STENOSIS_BINARY'
val_dataset = dict(
    type='CocoStenosisBinaryDataset',
    data_root='/ai/mnt/data/stenosis/selected/',
    ann_file='annotations/val_binary.json',
    data_prefix=dict(img='val/'))
warm_epoch = 50
train_ann_file = 'annotations/train_binary.json'
val_ann_file = 'annotations/val_binary.json'
train_data_prefix = dict(img='train/')
val_data_prefix = dict(img='val/')
launcher = 'none'
work_dir = '/ai/mnt/code/mmdetection/work_dirs/yolox_l_8xb8-1000e-lr5e-2_stenosis_binary_F0/best_coco_bbox_mAP_Show_epoch_300_test'
