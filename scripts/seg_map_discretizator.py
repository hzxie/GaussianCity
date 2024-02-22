# -*- coding: utf-8 -*-
#
# @File:   seg_map_discretizator.py
# @Author: Haozhe Xie
# @Date:   2023-12-25 15:52:37
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-22 19:51:11
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


def get_discrete_seg_maps(img):
    CLASSES = {
        # 0: NULL
        "NULL": torch.tensor([0, 0, 0], dtype=torch.int16, device=img.device),
        # 1: ROAD, FWY_DECK
        "ROAD": torch.tensor([230, 30, 30], dtype=torch.int16, device=img.device),
        # 2: FWY_PILLAR, FWY_BARRIER
        "FWY": torch.tensor([220, 220, 40], dtype=torch.int16, device=img.device),
        # 3: CAR
        "CAR": torch.tensor([20, 220, 40], dtype=torch.int16, device=img.device),
        # 4: WATER
        "WATER": torch.tensor([90, 215, 215], dtype=torch.int16, device=img.device),
        # 5: SKY
        "SKY": torch.tensor([20, 20, 20], dtype=torch.int16, device=img.device),
        # 6: ZONE
        "ZONE": torch.tensor([15, 15, 200], dtype=torch.int16, device=img.device),
        # 7: BLDG_FACADE
        "BLDG_FACADE": torch.tensor(
            [150, 105, 25], dtype=torch.int16, device=img.device
        ),
        # 8: BLDG_ROOF
        "BLDG_ROOF": torch.tensor([230, 50, 215], dtype=torch.int16, device=img.device),
    }
    h, w, _ = img.shape
    dists = torch.zeros((h, w, len(CLASSES)))
    for idx, mean_color in enumerate(CLASSES.values()):
        dists[..., idx] = torch.sum(torch.abs(img - mean_color), dim=2)

    dists = torch.reshape(dists, (h * w, len(CLASSES)))
    # The undefined small objects will be assigned to random classes
    return torch.argmin(dists, dim=1).reshape(h, w).cpu().numpy()


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
