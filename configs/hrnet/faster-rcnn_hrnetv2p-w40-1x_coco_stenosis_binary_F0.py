_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis_binary.py",
    "../_base_/schedules/schedule_300e.py",
    "../_base_/default_runtime.py",
]

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
    backbone=dict(
        _delete_=True,
        type="HRNet",
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(40),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(40, 80),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320),
            ),
        ),
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://msra/hrnetv2_w40"),
    ),
    neck=dict(
        _delete_=True, type="HRFPN", in_channels=[40, 80, 160, 320], out_channels=256
    ),
    roi_head=dict(bbox_head=dict(num_classes=1)),
)
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
