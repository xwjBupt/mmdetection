_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis_binary.py",
    "../_base_/schedules/schedule_1000e.py",
    "../_base_/default_runtime.py",
]


train_dataloader = dict(batch_size=4, num_workers=8)
val_dataloader = dict(
    batch_size=8,
    num_workers=16,
)

data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[144.5754766729963, 144.5754766729963, 144.5754766729963],
    std=[55.8710224233549, 55.8710224233549, 55.8710224233549],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)
model = dict(data_preprocessor=data_preprocessor)

vis_backends = [
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(project="CEREBRAL_STENOSIS_MMDETECTION", name="BASE_RUN_HODLER", group = "STENOSIS_BINARY"),
    )
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

base_lr = 0.05
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=3.5, norm_type=2),
)
