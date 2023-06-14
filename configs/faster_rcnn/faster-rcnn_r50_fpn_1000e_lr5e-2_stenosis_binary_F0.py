_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis_binary.py",
    "../_base_/schedules/schedule_1000e.py",
    "../_base_/default_runtime.py",
]
dataset_type = "CocoStenosisBinaryDataset"
data_root = "/ai/mnt/data/stenosis/selected/Binary/FOLD0/"
dataset_name = "STENOSIS_BINARY"
train_ann_file = "annotations/train_binary.json"
val_ann_file = "annotations/val_binary.json"
train_data_prefix = dict(img="train/")
val_data_prefix = dict(img="val/")
train_dataset = dict(
    type="CocoStenosisBinaryDataset",
    data_root="/ai/mnt/data/stenosis/selected/Binary/FOLD0/",
    ann_file="annotations/train_binary.json",
    data_prefix=dict(img="train/"),
)

train_dataloader = dict(batch_size=8, num_workers=8, dataset=train_dataset)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root="/ai/mnt/data/stenosis/selected/Binary/FOLD0/",
        ann_file="annotations/val_binary.json",
        data_prefix=dict(img="val/"),
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file="/ai/mnt/data/stenosis/selected/Binary/FOLD0/"
    + "annotations/val_binary.json"
)
test_evaluator = val_evaluator

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
        init_kwargs=dict(project="CEREBRAL_STENOSIS", name="BASE_RUN_HODLER"),
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
