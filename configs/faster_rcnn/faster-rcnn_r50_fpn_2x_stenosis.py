_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection_stenosis.py",
    "../_base_/schedules/schedule_2x.py",
    "../_base_/default_runtime.py",
]

wandb_tag = "BASE_RUN"
vis_backends = [
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(project="CEREBRAL_STENOSIS", name=wandb_tag),
    )
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
