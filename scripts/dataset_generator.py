# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-12 20:19:42
# @Email:  root@haozhexie.com

import argparse
import logging
import logging.config
import numpy as np
import os
import pickle
import scipy
import sys
import torch

from PIL import Image
from tqdm import tqdm


PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import footprint_extruder
import utils.helpers

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None


CLASSES = {
    "GAUSSIAN": {
        "NULL": 0,
        "ROAD": 1,
        "FWY_DECK": 1,
        "FWY_PILLAR": 2,
        "FWY_BARRIER": 2,
        "CAR": 3,
        "WATER": 4,
        "SKY": 5,
        "ZONE": 6,
        "BLDG_FACADE": 7,
        "BLDG_ROOF": 8,
    },
    "HOUDINI": {
        "ROAD": 1,
        "FWY_DECK": 2,
        "FWY_PILLAR": 3,
        "FWY_BARRIER": 4,
        "ZONE": 5,
    },
}

SCALES = {
    "ROAD": 10,
    "FWY_DECK": 10,
    "FWY_PILLAR": 5,
    "FWY_BARRIER": 2,
    "CAR": 1,
    "WATER": 20,
    "SKY": 20,
    "ZONE": 10,
    "BLDG_FACADE": 5,
    "BLDG_ROOF": 5,
}

CONSTANTS = {
    "MAP_SIZE": 24576,
    "BLDG_INS_MIN_ID": 100,
    "CAR_INS_MIN_ID": 5000,
}


def get_points_projection(points):
    car_rows = points[:, 3] >= CONSTANTS["CAR_INS_MIN_ID"]
    fwy_rows = np.isin(
        points[:, 3],
        [
            CLASSES["HOUDINI"]["FWY_DECK"],
            CLASSES["HOUDINI"]["FWY_PILLAR"],
            CLASSES["HOUDINI"]["FWY_BARRIER"],
        ],
    )
    rest_rows = ~np.logical_or(car_rows, fwy_rows)
    return {
        "CAR": _get_get_points_projection(points[car_rows]),
        "FWY": _get_get_points_projection(points[fwy_rows]),
        "REST": _get_get_points_projection(points[rest_rows]),
    }


def _get_get_points_projection(points):
    INVERSE_INDEX = {v: k for k, v in CLASSES["HOUDINI"].items()}
    pts_map = np.zeros((CONSTANTS["MAP_SIZE"], CONSTANTS["MAP_SIZE"]), dtype=bool)
    seg_map = np.zeros(
        (CONSTANTS["MAP_SIZE"], CONSTANTS["MAP_SIZE"]), dtype=points.dtype
    )
    tpd_hf = -1 * np.ones(
        (CONSTANTS["MAP_SIZE"], CONSTANTS["MAP_SIZE"]), dtype=points.dtype
    )
    btu_hf = np.iinfo(points.dtype).max * np.ones(
        (CONSTANTS["MAP_SIZE"], CONSTANTS["MAP_SIZE"]), dtype=points.dtype
    )
    for p in tqdm(points, leave=False):
        x, y, z, c_id = p
        c_name = INVERSE_INDEX[c_id] if c_id in INVERSE_INDEX else None
        # WARNING: The Scale for BLDG_ROOF would be the same as BLDG_FACADE.
        if c_name is None:
            if c_id < CONSTANTS["CAR_INS_MIN_ID"]:
                c_name = "BLDG_FACADE"
            else:
                c_name = "CAR"

        s = SCALES[c_name]
        x += CONSTANTS["MAP_SIZE"] // 2
        y += CONSTANTS["MAP_SIZE"] // 2
        pts_map[y, x] = True
        if tpd_hf[y, x] < z:
            tpd_hf[y : y + s, x : x + s] = z
            seg_map[y : y + s, x : x + s] = (
                CLASSES["GAUSSIAN"][c_name]
                if c_name not in ["BLDG_FACADE", "BLDG_ROOF", "CAR"]
                else c_id
            )
        if btu_hf[y, x] > z:
            btu_hf[y : y + s, x : x + s] = z

    return {"PTS": pts_map, "SEG": seg_map, "TD_HF": tpd_hf, "BU_HF": btu_hf}


def get_water_areas(projection):
    # The rest areas are assigned as the water areas, which is aligned with CitySample.
    water_area = projection["SEG"] == CLASSES["GAUSSIAN"]["NULL"]
    projection["SEG"][water_area] = CLASSES["GAUSSIAN"]["WATER"]
    projection["TD_HF"][water_area] = 0
    projection["BU_HF"][water_area] = 0

    wa_pts_map = _get_point_map(projection["PTS"].shape, SCALES["WATER"])
    wa_pts_map[~water_area] = False
    projection["PTS"] += wa_pts_map
    return projection


def _get_point_map(map_size, stride):
    pts_map = np.zeros(map_size, dtype=bool)
    ys = np.arange(0, map_size[0], stride)
    xs = np.arange(0, map_size[1], stride)
    coords = np.stack(np.meshgrid(ys, xs), axis=-1).reshape(-1, 2)
    pts_map[coords[:, 0], coords[:, 1]] = True
    return pts_map


def dump_projections(projections, output_dir, is_debug):
    os.makedirs(output_dir, exist_ok=True)
    for c, p in projections.items():
        for k, v in p.items():
            out_fpath = os.path.join(output_dir, "%s-%s.png" % (c, k))
            if is_debug:
                if k == "SEG":
                    utils.helpers.get_ins_seg_map(v).save(out_fpath)
                else:
                    Image.fromarray((v / np.max(v) * 255).astype(np.uint8)).save(
                        out_fpath
                    )
            else:
                Image.fromarray(v.astype(np.int16)).save(out_fpath)


def load_projections(output_dir):
    CATEGORIES = ["CAR", "FWY", "REST"]
    MAP_NAMES = ["SEG", "TD_HF", "BU_HF", "PTS"]

    projections = {}
    for c in CATEGORIES:
        if c not in projections:
            projections[c] = {}
        for m in MAP_NAMES:
            out_fpath = os.path.join(output_dir, "%s-%s.png" % (c, m))
            projections[c][m] = np.array(Image.open(out_fpath)).astype(np.int16)
            logging.debug(
                "Map[%s/%s] Max Value: %d Size: %s"
                % (c, m, np.max(projections[c][m]), projections[c][m].shape)
            )
    return projections


def get_points_from_projections(projections):
    points = np.empty((0, 5), dtype=np.int16)
    for c, p in projections.items():
        _points = _get_points_from_projection(p)
        points = np.concatenate((points, _points), axis=0)
        logging.debug("Category: %s: #Points: %d" % (c, len(_points)))

    logging.debug("#Points: %d" % (len(points)))
    return points


def _get_points_from_projection(projection):
    return footprint_extruder.get_points_from_projection(
        {v: k for k, v in CLASSES["GAUSSIAN"].items()},
        SCALES,
        projection["SEG"],
        projection["TD_HF"],
        projection["BU_HF"],
        projection["PTS"].astype(bool),
    )


def get_camera_poses(cam_pose, map_size):
    cam_pose["tx"] = float(cam_pose["tx"]) / 100 + map_size[1]
    cam_pose["ty"] = float(cam_pose["ty"]) / 100 + map_size[0]
    cam_pose["tz"] = float(cam_pose["tz"]) / 100
    cam_position = np.array([cam_pose["tx"], cam_pose["ty"], cam_pose["tz"]])
    cam_look_at = get_look_at_position(
        cam_position,
        np.array(
            [
                float(cam_pose["qx"]),
                float(cam_pose["qy"]),
                float(cam_pose["qz"]),
                float(cam_pose["qw"]),
            ]
        ),
    )
    return {
        "cam_position": cam_position,
        "cam_look_at": cam_look_at,
    }


def get_look_at_position(cam_position, cam_quaternion):
    mat3 = scipy.spatial.transform.Rotation.from_quat(cam_quaternion).as_matrix()
    return cam_position + mat3[:3, 0]


def main(data_dir, seg_map_file_pattern, gpus, is_debug):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    cities = sorted(os.listdir(data_dir))
    for city in tqdm(cities):
        city_dir = os.path.join(data_dir, city)
        points_file_path = os.path.join(city_dir, "POINTS.pkl")
        if not os.path.exists(points_file_path):
            logging.warning("File %s not found for %s" % (points_file_path, city))
            continue

        with open(points_file_path, "rb") as fp:
            points = pickle.load(fp)

        # NOTE: 5x means that the values are scaled up by a factor of 5 in Houdini export.
        logging.debug(
            "[5x] X Min: %d, X Max: %d" % (np.min(points[:, 0]), np.max(points[:, 0]))
        )
        logging.debug(
            "[5x] Y Min: %d, Y Max: %d" % (np.min(points[:, 1]), np.max(points[:, 1]))
        )
        logging.debug(
            "[5x] Z Min: %d, Z Max: %d" % (np.min(points[:, 2]), np.max(points[:, 2]))
        )
        logging.debug(
            "Building Max: %d, Car Max: %d"
            % (
                np.max(points[:, 3][points[:, 3] < CONSTANTS["CAR_INS_MIN_ID"]]),
                np.max(points[:, 3][points[:, 3] > CONSTANTS["CAR_INS_MIN_ID"]]),
            )
        )

        logging.info("Generating point projections...")
        projections = get_points_projection(points)

        logging.info("Generating water areas...")
        projections["REST"] = get_water_areas(projections["REST"])

        # logging.info("Saving projections...")
        proj_dir = os.path.join(city_dir, "Projection")
        dump_projections(projections, proj_dir, is_debug)

        # logging.info("loading projections...")
        # projections = load_projections(proj_dir)

        logging.info("Generate initial points...")
        points = get_points_from_projections(projections)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_HOME, "data"))
    parser.add_argument("--seg_map", default="SemanticImage/%sSequence.%04d.png")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.seg_map, args.gpu, args.debug)
