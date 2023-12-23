# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-12-23 13:11:16
# @Email:  root@haozhexie.com

import argparse
import csv
import json
import logging
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

import extensions.footprint_extruder
import extensions.topdown_projector
import extensions.voxlib
import utils.helpers

CONSTANTS = {
    "VOL_HEIGHT": 1536,
    "VOL_WIDTH": 1536,
    "VOL_DEPTH": 384,
    "DEPTH_OFFSET": 1,
    "STATIC_SCALE": 2,
}


def get_topdown_projection(points):
    tdp = extensions.topdown_projector.TopDownProjector()
    # NOTE: Semantic Labels
    # Road: 1; Freeway: 2; Others: 6
    # Pillar: 3; Water: 4, Sky: 5
    freeway_indexes = torch.isin(
        points[:, 3], torch.tensor([2, 3], device=points.device)
    )
    freeways = points[torch.where(freeway_indexes)]
    # Volume without freeways
    points = points[torch.where(~freeway_indexes)]
    volume = get_volume(points, scale=CONSTANTS["STATIC_SCALE"])
    with torch.no_grad():
        seg_map, height_field = tdp(volume.unsqueeze(dim=0))

    seg_map[seg_map == 0] = 4  # Assign the rest area as water area (as in City Sample)
    return seg_map.squeeze(dim=0), height_field.squeeze(dim=0), freeways


def get_volume(points, scale=1, volume=None):
    # Initialize an empty 3D volume
    if volume is None:
        volume = torch.zeros(
            (CONSTANTS["VOL_HEIGHT"], CONSTANTS["VOL_WIDTH"], CONSTANTS["VOL_DEPTH"]),
            dtype=points.dtype,
            device=points.device,
        )

    # Extract coordinates and values from the input tensor
    values = points[:, 3]
    coordinates = points[:, :3].float()
    coordinates[:, 0] = (coordinates[:, 0] + volume.size(1)) / scale  # x
    coordinates[:, 1] = (coordinates[:, 1] + volume.size(0)) / scale  # y
    coordinates[:, 2] = coordinates[:, 2] / scale + CONSTANTS["DEPTH_OFFSET"]  # z
    coordinates = (coordinates + 0.5).long()
    logging.debug("Volume Size (HxWxD): %s" % (volume.size(),))
    assert (coordinates[:, 0] < volume.size(1)).all()
    assert (coordinates[:, 1] < volume.size(0)).all()
    assert (coordinates[:, 2] < volume.size(2)).all()
    assert (coordinates[:, 2] >= 0).all()

    volume[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = values
    return volume


def get_volume_with_roof_1f(height_field, seg_map, freeways):
    fe = extensions.footprint_extruder.FootprintExtruder(
        max_height=CONSTANTS["VOL_DEPTH"]
    )
    logging.debug(
        "HF and Seg Map Shapes: %s, %s" % (height_field.size(), seg_map.size())
    )
    volume = fe(
        height_field.unsqueeze(dim=0).unsqueeze(dim=0),
        seg_map.unsqueeze(dim=0).unsqueeze(dim=0),
    )
    volume = get_volume(
        freeways, scale=CONSTANTS["STATIC_SCALE"], volume=volume.squeeze(dim=0)
    )
    return volume


def get_voxel_raycasting(cam_rig, cam_pose, volume):
    # TODO: Consider the scale for cars
    cam_pose["tx"] = (float(cam_pose["tx"]) / 100 + volume.size(1)) / CONSTANTS[
        "STATIC_SCALE"
    ]
    cam_pose["ty"] = (float(cam_pose["ty"]) / 100 + volume.size(0)) / CONSTANTS[
        "STATIC_SCALE"
    ]
    cam_pose["tz"] = (
        float(cam_pose["tz"]) / 100 / CONSTANTS["STATIC_SCALE"]
        + CONSTANTS["DEPTH_OFFSET"]
    )
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
    return get_ray_voxel_intersection(cam_rig, cam_position, cam_look_at, volume)


def get_look_at_position(cam_position, cam_quaternion):
    mat3 = scipy.spatial.transform.Rotation.from_quat(cam_quaternion).as_matrix()
    return cam_position + mat3[:3, 0]


def get_ray_voxel_intersection(cam_rig, cam_position, cam_look_at, volume):
    N_MAX_SAMPLES = 6
    cam_origin = torch.tensor(
        [
            cam_position[1],
            cam_position[0],
            cam_position[2],
        ],
        dtype=torch.float32,
        device=volume.device,
    )
    viewdir = torch.tensor(
        [
            cam_look_at[1] - cam_position[1],
            cam_look_at[0] - cam_position[0],
            cam_look_at[2] - cam_position[2],
        ],
        dtype=torch.float32,
        device=volume.device,
    )
    (
        voxel_id,
        depth2,
        raydirs,
    ) = extensions.voxlib.ray_voxel_intersection_perspective(
        volume,
        cam_origin,
        viewdir,
        torch.tensor([0, 0, 1], dtype=torch.float32),
        cam_rig["intrinsics"][0],
        [
            cam_rig["sensor_size"][1] / 2,
            cam_rig["sensor_size"][0] / 2,
        ],
        [cam_rig["sensor_size"][1], cam_rig["sensor_size"][0]],
        N_MAX_SAMPLES,
    )
    return {
        "voxel_id": voxel_id,
        "depth2": depth2,
        "raydirs": raydirs,
        "viewdir": viewdir,
        "cam_origin": cam_origin,
    }


def main(data_dir, is_debug):
    cities = sorted(os.listdir(data_dir))
    for city in tqdm(cities):
        with open(os.path.join(data_dir, city, "VOLUME.pkl"), "rb") as fp:
            volume = pickle.load(fp)

        logging.debug(
            "X Min: %d, X Max: %d" % (np.min(volume[:, 0]), np.max(volume[:, 0]))
        )
        logging.debug(
            "Y Min: %d, X Max: %d" % (np.min(volume[:, 1]), np.max(volume[:, 1]))
        )
        logging.debug(
            "Z Min: %d, X Max: %d" % (np.min(volume[:, 2]), np.max(volume[:, 2]))
        )
        logging.debug(
            "Label Min: %d, Label Max: %d"
            % (np.min(volume[:, 3]), np.max(volume[:, 3]))
        )
        assert np.max(volume) <= np.iinfo(np.int32).max

        # Compress the volume to semantic map, height field, and freeway 3D points
        volume = torch.from_numpy(volume.astype(np.int32)).cuda()
        seg_map, height_field, freeways = get_topdown_projection(volume)
        Image.fromarray(seg_map.cpu().numpy()).save(
            os.path.join(data_dir, city, "seg.png")
        )
        Image.fromarray(height_field.cpu().numpy()).save(
            os.path.join(data_dir, city, "hf.png")
        )
        with open(os.path.join(data_dir, city, "freeway.pkl"), "wb") as fp:
            pickle.dump(freeways, fp)

        # Rebuild 3D volume with roof and 1f
        volume = get_volume_with_roof_1f(height_field, seg_map, freeways)

        # Generate raycasting results
        raycasting_dir = os.path.join(data_dir, city, "raycasting")
        os.makedirs(raycasting_dir, exist_ok=True)
        with open(os.path.join(data_dir, city, "CameraRig.json")) as fp:
            cam_rig = json.load(fp)
        with open(os.path.join(data_dir, city, "CameraPoses.csv")) as fp:
            reader = csv.DictReader(fp)
            for row in tqdm(reader):
                raycasting = get_voxel_raycasting(
                    cam_rig["cameras"]["CameraComponent"], row, volume
                )
                if is_debug:
                    seg_map = utils.helpers.get_seg_map(
                        raycasting["voxel_id"].squeeze()[..., 0].cpu().numpy()
                    )
                    utils.helpers.get_diffuse_shading_img(
                        seg_map,
                        raycasting["depth2"],
                        raycasting["raydirs"],
                        raycasting["cam_origin"],
                    ).save(os.path.join(raycasting_dir, "%04d.png" % int(row["id"])))
                else:
                    with open(
                        os.path.join(raycasting_dir, "%04d.pkl" % int(row["id"])), "wb"
                    ) as ofp:
                        pickle.dump(raycasting, ofp)

                assert False


if __name__ == "__main__":
    logging.basicConfig(
        # filename=os.path.join(PROJECT_HOME, "output", "dataset-generator.log"),
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_HOME, "data"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.debug)
