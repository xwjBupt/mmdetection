# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
import glob
from tqdm import tqdm
import json
import torch
from loguru import logger
import shutil
import csv
import copy
from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()
        writer.writerow(content)


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument(
        "--config",
        default="/home/xwj/WORK/xcode/mmdetection/output_work_dirs/faster-rcnn_r50_fpn_300e_lr5e-2_stenosis_binary_F0/faster-rcnn_r50_fpn_300e_lr5e-2_stenosis_binary_F0.py",
        help="test config file path",
    )
    parser.add_argument(
        "--phase",
        default="test",
        choices=["train", "test"],
        type=str,
        help="use last checkpoints to test on train dataset, and the best checkpoints to test on test dataset",
    )
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show_dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def force_cudnn_initialization():
    s = 32
    dev = torch.device("cuda")
    a = torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )
    print(a)
    return a


force_cudnn_initialization()


def main():
    args = parse_args()
    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()
    os.environ["WANDB_MODE"] = "dryrun"
    torch.cuda.empty_cache()
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_name = osp.splitext(osp.basename(args.config))[0]
    config_dir = os.path.dirname(args.config)
    phase = args.phase
    logger.add(os.path.join(config_dir, "test.log"))
    root_cfg = Config.fromfile(args.config)
    dataset_name = root_cfg.dataset_name
    root_cfg.launcher = args.launcher
    if args.cfg_options is not None:
        root_cfg.merge_from_dict(args.cfg_options)
    logger.info(" <<< INFER START ON %s>>> " % config_dir)

    cfg = copy.deepcopy(root_cfg)
    # cfg._cfg_dict.visualizer.vis_backends = [dict(type="LocalVisBackend")]
    best_coco_bbox_mAP = -1
    best_coco_bbox_mAP_epoch = -1
    best_coco_bbox_result = {}

    if phase == "test":
        checkpoint = glob.glob(osp.join(config_dir, "*best*.pth"))[0]
    elif phase == "train":
        checkpoints = glob.glob(osp.join(config_dir, "*.pth"))
        max_epoch = max(
            [int(os.path.basename(i).split("_")[-1][:-4]) for i in checkpoints]
        )
        checkpoint = glob.glob(osp.join(config_dir, "*%d*.pth" % max_epoch))[0]
        cfg._cfg_dict.test_dataloader.dataset.ann_file = cfg._cfg_dict.train_ann_file
        cfg._cfg_dict.test_dataloader.dataset.data_prefix = (
            cfg._cfg_dict.train_data_prefix
        )
        cfg._cfg_dict.test_evaluator.ann_file = (
            cfg._cfg_dict.test_evaluator.ann_file.replace("val", "train")
        )

    else:
        assert False, "%s not support yet" % phase
    # load config
    epoch_name = os.path.basename(checkpoint).split("_")[-1][:-4]
    work_dir = checkpoint[:-4].replace("epoch_", "Show_epoch_") + "_%s" % phase
    cfg.work_dir = work_dir
    args.show_dir = work_dir
    cfg.load_from = checkpoint
    # if not args.show_dir:
    #     work_dir = checkpoint[:-4].replace("epoch_", "Show_epoch_") + "_%s" % phase
    #     cfg.work_dir = work_dir
    # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get("work_dir", None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     # cfg.work_dir = osp.join(project_root + "/work_dirs", config_name)
    #     cfg.work_dir = args.show_dir
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    if args.tta:
        if "tta_model" not in cfg:
            warnings.warn(
                "Cannot find ``tta_model`` in config, " "we will set it as default."
            )
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            )
        if "tta_pipeline" not in cfg:
            warnings.warn(
                "Cannot find ``tta_pipeline`` in config, " "we will set it as default."
            )
            test_data_cfg = cfg.test_dataloader.dataset
            while "dataset" in test_data_cfg:
                test_data_cfg = test_data_cfg["dataset"]
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [
                        dict(
                            type="PackDetInputs",
                            meta_keys=(
                                "img_id",
                                "img_path",
                                "ori_shape",
                                "img_shape",
                                "scale_factor",
                                "flip",
                                "flip_direction",
                            ),
                        )
                    ],
                ],
            )
            cfg.tta_pipeline[-1] = flip_tta
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

        # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()

    ### record result ###
    result_json = glob.glob(os.path.join(work_dir, "*", "*.json"))[0]
    with open(result_json, "r") as load_f:
        result_dict = json.load(load_f)
    if result_dict.get("coco/bbox_mAP") > best_coco_bbox_mAP:
        best_coco_bbox_mAP = result_dict.get("coco/bbox_mAP")
        best_coco_bbox_mAP_epoch = epoch_name
        best_coco_bbox_result_dict = result_dict
        best_coco_bbox_result_dict["Method"] = checkpoint.split("output_work_dirs")[-1]
        best_coco_bbox_result_dict["Phase"] = phase
    print("\n\n")
    csvname = os.path.join(project_root, dataset_name + ".csv")
    write_to_csv(
        csvname,
        best_coco_bbox_result_dict,
    )
    logger.info(
        "FINAL best coco_bbox_mAP {} @ Epoch {} with result as {} to ".format(
            best_coco_bbox_mAP,
            best_coco_bbox_mAP_epoch,
            best_coco_bbox_result_dict,
            csvname,
        )
    )
    del runner
    logger.info(" <<< INFER DONE ON %s>>> \n\n" % config_dir)


if __name__ == "__main__":
    main()
