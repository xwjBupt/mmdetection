_base_ = "./yolox_l_8xb8-1000e-lr5e-2_stenosis_binary.py"


dataset_type = "CocoStenosisBinaryDataset"
data_root = "/ai/mnt/data/stenosis/selected/Binary/FOLD2/"
dataset_name = "STENOSIS_BINARY"
train_ann_file = "annotations/train_binary.json"
val_ann_file = "annotations/val_binary.json"
train_data_prefix = dict(img="train/")
val_data_prefix = dict(img="val/")

train_dataset = dict(
    dataset=dict(
        type="CocoStenosisBinaryDataset",
        data_root="/ai/mnt/data/stenosis/selected/Binary/FOLD2/",
        ann_file="annotations/train_binary.json",
        data_prefix=dict(img="train/"),
    )
)

train_dataloader = dict(batch_size=8, num_workers=4, dataset=train_dataset)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root="/ai/mnt/data/stenosis/selected/Binary/FOLD2/",
        ann_file="annotations/val_binary.json",
        data_prefix=dict(img="val/"),
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file="/ai/mnt/data/stenosis/selected/Binary/FOLD2/"
    + "annotations/val_binary.json"
)
test_evaluator = val_evaluator
