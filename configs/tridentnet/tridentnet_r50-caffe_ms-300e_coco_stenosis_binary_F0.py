_base_ = [
    "../_base_/models/faster-rcnn_r50-caffe-c4.py",
    "../_base_/datasets/coco_detection_stenosis_binary.py",
    "../_base_/schedules/schedule_300e.py",
    "../_base_/default_runtime.py",
]

data_preprocessor = dict(
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
)

model = dict(
    type="TridentFasterRCNN",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="TridentResNet",
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    ),
    roi_head=dict(
        type="TridentRoIHead",
        num_branch=3,
        test_branch_idx=1,
        bbox_head=dict(num_classes=1),
    ),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=500),
        rcnn=dict(sampler=dict(num=128, pos_fraction=0.5, add_gt_as_proposals=False)),
    ),
)


train_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomChoiceResize",
        scales=[
            (1333, 640),
            (1333, 672),
            (1333, 704),
            (1333, 736),
            (1333, 768),
            (1333, 800),
        ],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline), batch_size=4, num_workers=8
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
)
