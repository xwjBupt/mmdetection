# Copyright (c) OpenMMLab. All rights reserved.
# This file add snake case alias for coco api

import warnings
from collections import defaultdict
from typing import List, Optional, Union
import time
import numpy as np
import copy
import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval
import pycocotools.mask as maskUtils


class COCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(self, annotation_file=None):
        if getattr(pycocotools, "__version__", "0") >= "12.0.2":
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning,
            )
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


# just for the ease of import
# COCOeval = _COCOeval
class COCOeval(_COCOeval):
    # def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
    #     super().__init__()

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(
                "useSegm (deprecated) is not None. Running {} evaluation".format(
                    p.iouType
                )
            )
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            # computeIoU = self.computeIoU
            computeIoU = self.cin_compute_iou
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def cin_compute_iou(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        if p.box_contain_thresh < 0.5:
            ious = self.update_ious(d, g, ious, box_contain_thresh=p.box_contain_thresh)
        return ious

    def update_ious(self, d, g, ious, box_contain_thresh=0.1):
        raw_ious = copy.deepcopy(ious)
        for dindex, di in enumerate(d):
            for gindex, gi in enumerate(g):
                dx, dy, dw, dh = di
                gx, gy, gw, gh = gi
                g_center_x = gx + gw / 2
                g_center_y = gy + gh / 2
                d_center_x = dx + dw / 2
                d_center_y = dy + dh / 2

                bing = (
                    g_center_x > dx
                    and g_center_x < (dx + dw)
                    and g_center_y > dy
                    and g_center_y < (dy + dh)
                )
                ginb = (
                    d_center_x > gx
                    and d_center_x < (gx + gw)
                    and d_center_y > gy
                    and d_center_y < (gy + gh)
                )
                gcontainb = (
                    gx > dx
                    and gy > dy
                    and (gx + gw) > (dx + dw)
                    and (gy + gh) > (dy + dh)
                )
                # TODO keyfunction
                if bing and ginb:
                    if gcontainb:
                        ious[dindex][gindex] = 0.95
                    else:
                        if ious[dindex][gindex] > box_contain_thresh:
                            ious[dindex][gindex] = max(0.5, ious[dindex][gindex])

                # if bing or ginb:
                #     ious[dindex][gindex] = 0.75

                # if (
                #     g_center_x > dx
                #     and g_center_x < (dx + dw)
                #     and g_center_y > dy
                #     and g_center_y < (dy + dh)
                # ):  # gt_center in bbox
                #     if (
                #         d_center_x > gx
                #         and d_center_x < (gx + gw)
                #         and d_center_y > gy
                #         and d_center_y < (gy + gh)
                #     ):  # bbox_center in gt box
                #         if (
                #             gx > dx
                #             and gy > dy
                #             and (gx + gw) > (dx + dw)
                #             and (gy + gh) > (dy + dh)
                #         ):
                #             ious[dindex][gindex] = 0.95
                #             print("fsdfsdfsd")
                # if (
                #     g_center_x > dx
                #     and g_center_x < (dx + dw)
                #     and g_center_y > dy  # gt_center in gt bbox
                #     and g_center_y < (dy + dh)
                # ) or (
                #     d_center_x > gx
                #     and d_center_x < (gx + gw)
                #     and d_center_y > gy  # bbox_center in gt
                #     and d_center_y < (gy + gh)
                # ):
                #     if ious[dindex][gindex] > 0.2:
                #         ious[dindex][gindex] = max(0.5, ious[dindex][gindex])
                #     print("fsdfsdfsd")
        return ious


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str, optional): Path of annotation file.
            Defaults to None.
    """

    def __init__(self, annotation_file: Optional[str] = None) -> None:
        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self) -> None:
        """Create index."""
        # create index
        print("creating index...")
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                for seg_ann in ann["segments_info"]:
                    # to match with instance.json
                    seg_ann["image_id"] = ann["image_id"]
                    img_to_anns[ann["image_id"]].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    # annotations from different images but
                    # may have same segment_id
                    if seg_ann["id"] in anns.keys():
                        anns[seg_ann["id"]].append(seg_ann)
                    else:
                        anns[seg_ann["id"]] = [seg_ann]

            # filter out annotations from other images
            img_to_anns_ = defaultdict(list)
            for k, v in img_to_anns.items():
                img_to_anns_[k] = [x for x in v if x["image_id"] == k]
            img_to_anns = img_to_anns_

        if "images" in self.dataset:
            for img_info in self.dataset["images"]:
                img_info["segm_file"] = img_info["file_name"].replace("jpg", "png")
                imgs[img_info["id"]] = img_info

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                for seg_ann in ann["segments_info"]:
                    cat_to_imgs[seg_ann["category_id"]].append(ann["image_id"])

        print("index created!")

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self, ids: Union[List[int], int] = []) -> Optional[List[dict]]:
        """Load anns with the specified ids.

        ``self.anns`` is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (Union[List[int], int]): Integer ids specifying anns.

        Returns:
            anns (List[dict], optional): Loaded ann objects.
        """
        anns = []

        if hasattr(ids, "__iter__") and hasattr(ids, "__len__"):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]
