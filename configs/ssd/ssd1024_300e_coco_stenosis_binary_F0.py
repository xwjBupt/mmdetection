_base_ = [
    "../_base_/models/ssd300.py",
    "../_base_/datasets/coco_detection_stenosis_binary.py",
    "../_base_/schedules/schedule_300e.py",
    "../_base_/default_runtime.py",
]

# dataset settings
input_size = 1024
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            144.5754766729963,
            144.5754766729963,
            144.5754766729963,
        ],
        pad_size_divisor=32,
        std=[
            55.8710224233549,
            55.8710224233549,
            55.8710224233549,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=4,
    ),
    bbox_head=dict(
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        ),
        num_classes=1,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Expand",
        mean=[
            144.5754766729963,
            144.5754766729963,
            144.5754766729963,
        ],
        to_rgb=True,
        ratio_range=(1, 4),
    ),
    dict(
        type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3
    ),
    dict(type="Resize", scale=(input_size, input_size), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="Resize", scale=(input_size, input_size), keep_ratio=False),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type="CocoStenosisBinaryDataset",
            data_root="/home/xwj/Xdata/stenosis/selected/Binary/FOLD0/COCO/",
            ann_file="annotations/train_binary.json",
            data_prefix=dict(img="train/"),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args={{_base_.backend_args}},
        ),
    ),
)
val_dataloader = dict(batch_size=16, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=2e-3, momentum=0.9, weight_decay=5e-4),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="CheckInvalidLossHook", interval=50, priority="VERY_LOW"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(
            init_kwargs=dict(
                group="STENOSIS_BINARY",
                name="BASE_RUN_HODLER",
                project="CEREBRAL_STENOSIS_MMDETECTION",
            ),
            type="WandbVisBackend",
        ),
    ],
)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
