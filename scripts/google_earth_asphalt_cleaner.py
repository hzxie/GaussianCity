# -*- coding: utf-8 -*-
#
# @File:   google_earth_asphalt_cleaner.py
# @Author: Haozhe Xie
# @Date:   2023-07-03 09:49:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-04-05 10:00:48
# @Email:  root@haozhexie.com

import argparse
import logging
import numpy as np
import pickle
import os

from PIL import Image
from tqdm import tqdm


def main(ges_dir, asphalt_img):
    asphalt_img = np.array(Image.open(asphalt_img))
    ah, aw, _ = asphalt_img.shape
    ges_projects = sorted(os.listdir(ges_dir))
    for gp in tqdm(ges_projects, leave=True):
        footage_dir = os.path.join(ges_dir, gp, "footage")
        ins_seg_dir = os.path.join(ges_dir, gp, "InstanceImage")
        pts_dir = os.path.join(ges_dir, gp, "Points")
        if not os.path.exists(ins_seg_dir):
            logging.warning(
                "Skip Project %s. No instance segmentation found in %s"
                % (gp, ins_seg_dir)
            )
            continue
        if not os.path.exists(pts_dir):
            logging.warning(
                "Skip Project %s. No initial points found in %s" % (gp, pts_dir)
            )
            continue

        footages = sorted(os.listdir(footage_dir))
        ins_segs = sorted(os.listdir(ins_seg_dir))
        points = sorted(os.listdir(pts_dir))
        for f, i, p in zip(footages, ins_segs, points):
            footage = np.array(Image.open(os.path.join(footage_dir, f)))
            ins_seg = np.array(Image.open(os.path.join(ins_seg_dir, i)))
            with open(os.path.join(pts_dir, p), "rb") as fp:
                pts = pickle.load(fp)

            fh, fw, _ = footage.shape
            # ROAD_ID == 1
            road_mask = (ins_seg == 1)[..., None].astype(np.uint8)
            y, x = np.random.randint(0, ah - fh), np.random.randint(0, aw - fw)
            _asphalt_img = asphalt_img[y : y + fh, x : x + fw]
            footage = _asphalt_img * road_mask + footage * (1 - road_mask)
            Image.fromarray(footage).save(os.path.join(footage_dir, f))
            # Update the mask for roads
            pts["msk"][road_mask[..., 0]] = 1
            with open(os.path.join(pts_dir, p), "wb") as fp:
                pickle.dump(pts, fp)


if __name__ == "__main__":
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ges_dir", default=os.path.join(PROJECT_HOME, "data", "google-earth")
    )
    parser.add_argument(
        "--asphalt_img", default=os.path.join(PROJECT_HOME, "output", "asphalt.jpg")
    )
    args = parser.parse_args()
    main(args.ges_dir, args.asphalt_img)
