_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis_binary.py",
    "../_base_/schedules/schedule_300e.py",
    "../_base_/default_runtime.py",
]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[144.5754766729963, 144.5754766729963, 144.5754766729963],
    std=[55.8710224233549, 55.8710224233549, 55.8710224233549],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5),
    bbox_head=dict(num_classes=1),
)

# optimizer
train_dataloader = dict(batch_size=4, num_workers=8)
val_dataloader = dict(batch_size=4, num_workers=4)

vis_backends = [
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="CEREBRAL_STENOSIS_MMDETECTION",
            name="BASE_RUN_HODLER",
            group="STENOSIS_BINARY",
        ),
    )
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
