_base_ = "./yolox_s_8xb8-300e_coco.py"

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256),
)


dataset_type = "CocoStenosisDataset"
data_root = "/ai/mnt/data/stenosis/selected/"
dataset_name = "STENOSIS"


train_dataset = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="train/"),
    )
)

train_dataloader = dict(batch_size=24, num_workers=8, dataset=train_dataset)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="annotations/val.json",
    data_prefix=dict(img="val/"),
)

val_dataloader = dict(batch_size=32, num_workers=8, dataset=val_dataset)
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator
