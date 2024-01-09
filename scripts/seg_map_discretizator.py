# -*- coding: utf-8 -*-
#
# @File:   seg_map_discretizator.py
# @Author: Haozhe Xie
# @Date:   2023-12-25 15:52:37
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-09 15:14:51
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
        "Undefined": torch.tensor(
            [255, 255, 255], dtype=torch.int16, device=img.device
        ),
        "Road": torch.tensor([255, 84, 50], dtype=torch.int16, device=img.device),
        "Freeway": torch.tensor([230, 235, 90], dtype=torch.int16, device=img.device),
        "Car": torch.tensor([60, 230, 110], dtype=torch.int16, device=img.device),
        "Water": torch.tensor([140, 230, 230], dtype=torch.int16, device=img.device),
        "Sky": torch.tensor([0, 0, 0], dtype=torch.int16, device=img.device),
        "Ground": torch.tensor([90, 110, 240], dtype=torch.int16, device=img.device),
        "Building": torch.tensor([180, 140, 30], dtype=torch.int16, device=img.device),
        "Roof": torch.tensor([250, 150, 240], dtype=torch.int16, device=img.device),
    }
    h, w, _ = img.shape
    dists = torch.zeros((h, w, len(CLASSES)))
    for idx, mean_color in enumerate(CLASSES.values()):
        dists[..., idx] = torch.sum(torch.abs(img - mean_color), dim=2)

    dists = torch.reshape(dists, (h * w, len(CLASSES)))
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
