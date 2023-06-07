_base_ = "./yolox_l_8xb8-300e_stenosis.py"

# training settings
max_epochs = 1000
num_last_epochs = 50
interval = 10
warm_epoch = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.05
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type="mmdet.QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=warm_epoch,
        convert_to_iter_based=True,
    ),
    dict(
        # use cosine lr from 5 to 285 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=warm_epoch,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # use fixed lr during last 15 epochs
        type="ConstantLR",
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]
