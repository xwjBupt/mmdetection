_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis.py",
    "../_base_/schedules/schedule_500e.py",
    "../_base_/default_runtime.py",
]

data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[144.5754766729963, 144.5754766729963, 144.5754766729963],
    std=[55.8710224233549, 55.8710224233549, 55.8710224233549],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)
model = dict(data_preprocessor=data_preprocessor)

# data_preprocessor = dict(
#     type="DetDataPreprocessor",
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_size_divisor=1,
#     pad_mask=True,
#     mask_pad_value=0,
#     pad_seg=True,
#     seg_pad_value=255,
# )

vis_backends = [
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(project="CEREBRAL_STENOSIS", name="BASE_RUN_HODLER"),
    )
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
