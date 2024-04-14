# -*- coding: utf-8 -*-
#
# @File:   footage_roof_cleaner.py
# @Author: Haozhe Xie
# @Date:   2023-07-05 09:59:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-04-14 10:10:23
# @Email:  root@haozhexie.com

import argparse
import cv2
import logging
import multiprocessing
import numpy as np
import os
import pickle
import random
import scipy.cluster

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

# CONSTANTS
CONSTANTS = {
    "N_COLOR_CLUSTERS": 5,
    "DEFAULT_ROOF_COLOR": 128,
    "MIN_BLDG_INS": 100,
}


def downsample_texture(roof_img_file):
    texture = np.array(Image.open(roof_img_file))
    texture = cv2.GaussianBlur(texture.astype(np.uint8), (0, 0), sigmaX=3, sigmaY=3)
    texture = cv2.resize(
        np.tile(texture, (48, 48, 1)), (texture.shape[1], texture.shape[0])
    )
    texture = np.tile(texture, (8, 8, 1))
    gray_texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray_texture[..., None].repeat(3, axis=2)


def _get_roof_colors(footage, bev_seg_map, roofs):
    DEFAULT_ROOF_COLOR = np.array((CONSTANTS["DEFAULT_ROOF_COLOR"],) * 3)
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k + 1.
    instances = [
        i
        for i in np.unique(bev_seg_map)
        if i > CONSTANTS["MIN_BLDG_INS"] and i % 2 == 0
    ]
    for i in instances:
        if i not in roofs:
            roofs[i] = []
        # Determine the dominant color of facade
        facade_mask = bev_seg_map == i
        building = footage.copy()
        building[~facade_mask[..., None].repeat(3, axis=2)] = 0

        fx, fy, fw, fh = cv2.boundingRect(cv2.findNonZero(facade_mask.astype(np.uint8)))
        dominant_color = DEFAULT_ROOF_COLOR
        if fx != 0 or fh != 0:
            building = building[fy : fy + fh, fx : fx + fw]
            building = cv2.resize(building, (16, 16))
            colors = building.reshape(-1, 3).astype(np.float32)
            codebook, _ = scipy.cluster.vq.kmeans(colors, CONSTANTS["N_COLOR_CLUSTERS"])
            code, _ = scipy.cluster.vq.vq(colors, codebook)
            counts, _ = np.histogram(code, len(codebook))
            while (dominant_color <= CONSTANTS["DEFAULT_ROOF_COLOR"]).all() and (
                counts != -1
            ).any():
                max_idx = np.argmax(counts)
                counts[max_idx] = -1
                dominant_color = codebook[max_idx]
        # Record the dominant color for the building
        if (dominant_color > CONSTANTS["DEFAULT_ROOF_COLOR"]).any():
            roofs[i].append(dominant_color)

    return roofs


def _get_roof_texture(roof_colors, roof_textures):
    DEFAULT_ROOF_COLOR = np.array((CONSTANTS["DEFAULT_ROOF_COLOR"],) * 3)
    roof_texture = random.choice(roof_textures).copy()
    if not roof_colors:
        roof_color = DEFAULT_ROOF_COLOR
    else:
        codebook, _ = scipy.cluster.vq.kmeans(
            np.array(roof_colors), min(CONSTANTS["N_COLOR_CLUSTERS"], len(roof_colors))
        )
        code, _ = scipy.cluster.vq.vq(roof_colors, codebook)
        counts, _ = np.histogram(code, len(codebook))
        roof_color = codebook[np.argmax(counts)]

    return (roof_texture * roof_color).astype(np.uint8)


def _get_cleaned_roof_footage(footage, bev_seg_map, roofs):
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k + 1.
    instances = [
        i
        for i in np.unique(bev_seg_map)
        if i > CONSTANTS["MIN_BLDG_INS"] and i % 2 == 0
    ]
    for i in instances:
        roof_mask = (bev_seg_map == i + 1)[..., None]
        roof_texture = roofs[i]
        rh, rw, _ = roof_texture.shape
        # Apply random affine transformation
        rot_mtrx = cv2.getRotationMatrix2D((rw // 2, rh // 2), random.randint(0, 45), 1)
        roof_texture = cv2.warpAffine(
            roof_texture, rot_mtrx, (rh, rw), borderMode=cv2.BORDER_REPLICATE
        )
        rh, rw, _ = roof_texture.shape
        mx, my, mw, mh = cv2.boundingRect(cv2.findNonZero(roof_mask.astype(np.uint8)))
        ry, rx = rh // 2 - mh // 2, rw // 2 - mw // 2
        assert rh > mh and rw > mw, (rh, mh, rw, mw)
        roof_img = np.zeros(footage.shape, dtype=np.uint8)
        roof_img[my : my + mh, mx : mx + mw] = roof_texture[ry : ry + mh, rx : rx + mw]

        # Make the border of roof more smooth
        # roof_mask = cv2.GaussianBlur(
        #     roof_mask.astype(np.uint8), (0, 0), sigmaX=3, sigmaY=3
        # )
        # roof_mask = cv2.dilate(roof_mask, np.ones((7, 7), dtype=np.uint8))
        # roof_mask = cv2.erode(roof_mask, np.ones((7, 7), dtype=np.uint8))
        # roof_mask = cv2.GaussianBlur(roof_mask, (0, 0), sigmaX=1, sigmaY=1)
        # roof_mask = roof_mask[..., None]
        # Replace roof textures
        footage = footage * (1 - roof_mask) + roof_img * roof_mask

    return footage


def replace_roof_texture(footage_dir, ins_seg_dir, roof_textures):
    if not os.path.exists(ins_seg_dir):
        logging.warning("No instance segmentation found in %s" % ins_seg_dir)
        return
    # if not os.path.exists(pts_dir):
    #     logging.warning("No initial points found in %s" % pts_dir)
    #     return

    # IO cache
    footage_files = sorted(os.listdir(footage_dir))
    ins_seg_files = sorted(os.listdir(ins_seg_dir))
    # points_files = sorted(os.listdir(pts_dir))
    footages = []
    ins_segs = []
    # points = []
    for f, i in zip(footage_files, ins_seg_files):
        footages.append(np.array(Image.open(os.path.join(footage_dir, f))))
        ins_segs.append(np.array(Image.open(os.path.join(ins_seg_dir, i))))
        # with open(os.path.join(pts_dir, p), "rb") as fp:
        #     points.append(pickle.load(fp))

    # Determine the roof colors for buildings in the current trajectory
    roofs = {}
    for footage, ins_seg in zip(footages, ins_segs):
        roofs = _get_roof_colors(footage, ins_seg, roofs)

    # Recolorize the roof texture with the facade dominant color
    for ins, colors in roofs.items():
        roofs[ins] = _get_roof_texture(colors, roof_textures)

    # Replace the roofs with cleaner ones
    for filename, footage, ins_seg in zip(footage_files, footages, ins_segs):
        footage = _get_cleaned_roof_footage(footage, ins_seg, roofs)
        Image.fromarray(footage.astype(np.uint8)).save(
            os.path.join(footage_dir, filename)
        )


def main(ges_dir, roof_img_dir, city="US-NewYork", n_processes=8):
    roof_files = [os.path.join(roof_img_dir, rf) for rf in os.listdir(roof_img_dir)]
    with multiprocessing.Pool() as pool:
        roof_textures = list(
            tqdm(
                pool.imap(downsample_texture, roof_files),
                total=len(roof_files),
            )
        )

    ges_projects = sorted([gp for gp in os.listdir(ges_dir) if gp.startswith(city)])
    # Single-threaded
    for gp in tqdm(ges_projects, leave=True):
        replace_roof_texture(
            os.path.join(ges_dir, gp, "footage"),
            os.path.join(ges_dir, gp, "InstanceImage"),
            roof_textures,
        )
    # Multi-threaded
    # with multiprocessing.Pool(n_processes) as pool:
    #     args = zip(
    #         [os.path.join(ges_dir, gp, "footage") for gp in ges_projects],
    #         [os.path.join(ges_dir, gp, "InstanceImage") for gp in ges_projects],
    #         [roof_textures for _ in ges_projects],
    #     )
    #     pool.starmap(
    #         replace_roof_texture,
    #         tqdm(
    #             args,
    #             total=len(ges_projects),
    #         ),
    #     )


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
        "--roof_img_dir", default=os.path.join(PROJECT_HOME, "output", "roofs")
    )
    args = parser.parse_args()
    main(args.ges_dir, args.roof_img_dir)
