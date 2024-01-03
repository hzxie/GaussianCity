# -*- coding: utf-8 -*-
#
# @File:   seg_map_discretizator.py
# @Author: Haozhe Xie
# @Date:   2023-12-25 15:52:37
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-12-26 16:19:11
# @Email:  root@haozhexie.com

import argparse
import logging
import numpy as np
import os
import sys

from tqdm import tqdm
from PIL import Image


PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.helpers


def get_discrete_seg_maps(img):
    CLASSES = {
        "Others": np.array([255, 255, 255]),
        "Road": np.array([255, 84, 50]),
        "Freeway": np.array([230, 235, 90]),
        "Car": np.array([60, 230, 110]),
        "Water": np.array([140, 230, 230]),
        "Sky": np.array([0, 0, 0]),
        "Building": np.array([180, 140, 30]),
        "Roof": np.array([250, 150, 240]),
        "Ground": np.array([90, 110, 240]),
    }
    h, w, _ = img.shape
    dists = np.zeros((h, w, len(CLASSES)))
    for idx, mean_color in enumerate(CLASSES.values()):
        dists[..., idx] = np.sum(np.abs(img - mean_color), axis=2)

    dists = np.reshape(dists, (h * w, len(CLASSES)))
    return np.argmin(dists, axis=1).reshape(h, w)


def main(input_dir, output_dir):
    images = sorted(os.listdir(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(images):
        img = Image.open(os.path.join(input_dir, i))
        seg_map = get_discrete_seg_maps(np.array(img))
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
    parser.add_argument("--output_dir", default="seg")
    args = parser.parse_args()
    main(
        os.path.join(args.work_dir, args.input_dir),
        os.path.join(args.work_dir, args.output_dir),
    )
