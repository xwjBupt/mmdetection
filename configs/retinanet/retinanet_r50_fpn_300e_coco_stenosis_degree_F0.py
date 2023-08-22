_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis_degree.py",
    "../_base_/schedules/schedule_300e.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(batch_size=8, num_workers=16)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
)
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[144.5754766729963, 144.5754766729963, 144.5754766729963],
    std=[55.8710224233549, 55.8710224233549, 55.8710224233549],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)
model = dict(data_preprocessor=data_preprocessor, bbox_head=dict(num_classes=3))
# learning rate policy
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1,
    ),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(
            init_kwargs=dict(
                group="STENOSIS_DEGREE",
                name="BASE_RUN_HODLER",
                project="CEREBRAL_STENOSIS_MMDETECTION",
            ),
            type="WandbVisBackend",
        ),
    ],
)
# optimizer
optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
)
