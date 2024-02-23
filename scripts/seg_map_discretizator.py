# -*- coding: utf-8 -*-
#
# @File:   seg_map_discretizator.py
# @Author: Haozhe Xie
# @Date:   2023-12-25 15:52:37
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-23 10:42:23
# @Email:  root@haozhexie.com

import argparse
import logging
import numpy as np
import os
import sys
import torch

from tqdm import tqdm
from PIL import Image


PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.helpers


def _get_tensor(value, device):
    return torch.tensor(value, dtype=torch.int16, device=device)


def get_discrete_seg_maps(img):
    CLASSES = {
        # 0: NULL
        _get_tensor([0, 0, 0], img.device): 0,
        _get_tensor([200, 200, 200], img.device): 0,
        # 1: ROAD, FWY_DECK
        _get_tensor([210, 5, 20], img.device): 1,
        _get_tensor([155, 0, 10], img.device): 1,
        # 2: FWY_PILLAR, FWY_BARRIER
        _get_tensor([220, 220, 40], img.device): 2,
        # _get_tensor([170, 170, 5], img.device): 2,
        # 3: CAR
        _get_tensor([20, 220, 40], img.device): 3,
        _get_tensor([0, 170, 0], img.device): 3,
        # 4: WATER
        _get_tensor([0, 160, 160], img.device): 4,
        _get_tensor([50, 200, 200], img.device): 4,
        # 5: SKY
        _get_tensor([10, 10, 10], img.device): 5,
        # 6: ZONE
        _get_tensor([15, 15, 200], img.device): 6,
        _get_tensor([0, 0, 150], img.device): 6,
        # 7: BLDG_FACADE
        _get_tensor([150, 105, 25], img.device): 7,
        # _get_tensor([170, 170, 15], img.device): 7,
        _get_tensor([120, 80, 5], img.device): 7,
        # 8: BLDG_ROOF
        _get_tensor([230, 60, 215], img.device): 8,
        _get_tensor([160, 0, 160], img.device): 8,
    }
    h, w, _ = img.shape
    dists = torch.zeros((h, w, len(CLASSES)))
    for idx, mean_color in enumerate(CLASSES.keys()):
        dists[..., idx] = torch.sum(torch.abs(img - mean_color), dim=2)

    dists = torch.reshape(dists, (h * w, len(CLASSES)))
    min_idx = torch.argmin(dists, dim=1).reshape(h, w).cpu().numpy()
    class_id = np.array([class_id for class_id in CLASSES.values()])
    return class_id[min_idx]


def main(input_dir, output_dir):
    images = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpeg")])
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(images):
        img = Image.open(os.path.join(input_dir, i))
        # NOTE: Replacing np.int16 to np.uint8 causes bugs in PyTorch
        img = torch.from_numpy(np.array(img).astype(np.int16)).cuda()
        seg_map = get_discrete_seg_maps(img)
        fn, _ = os.path.splitext(i)
        utils.helpers.get_seg_map(seg_map).save(os.path.join(output_dir, "%s.png" % fn))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir", default=os.path.join(PROJECT_HOME, "data", "City01")
    )
    parser.add_argument("--input_dir", default="SemanticImage")
    parser.add_argument("--output_dir", default="SemanticImage")
    args = parser.parse_args()
    main(
        os.path.join(args.work_dir, args.input_dir),
        os.path.join(args.work_dir, args.output_dir),
    )
