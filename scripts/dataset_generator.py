# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-20 18:56:32
# @Email:  root@haozhexie.com

import argparse
import cv2
import csv
import json
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

import extensions.footprint_extruder
import extensions.topdown_projector
import extensions.voxlib
import utils.helpers

CONSTANTS = {
    "VOL_HEIGHT": 1536,
    "VOL_WIDTH": 1536,
    "VOL_DEPTH": 384,
    "VOL_DEPTH_OFFSET": 1,
    "CAM_DEPTH_OFFSET": 3,
    "STATIC_SCALE": 2,
    "BLD_INS_MIN_ID": 100,
    "CAR_INS_MIN_ID": 5000,
    "CAR_CLS_ID": 3,
    "BLD_FACADE_CLS_ID": 7,
    "BLD_ROOF_CLS_ID": 8,
}


def get_topdown_projection(points):
    tdp = extensions.topdown_projector.TopDownProjector()
    # NOTE: Instance Labels
    #       In the export points from Houdini, building starts from 100 and
    #       car starts from 5000.
    #
    # Undefined: 0; Road: 1; Freeway: 2;
    # Car: 3; Water: 4; Sky: 5; Others (Ground): 6;
    # Building: 4n -> 7; Roof: 4n+1 -> 8
    freeway_indexes = torch.isin(points[:, 3], torch.tensor([2], device=points.device))
    freeways = points[torch.where(freeway_indexes)]
    # Volume without freeways
    points = points[torch.where(~freeway_indexes)]
    volume = get_volume(points, scale=CONSTANTS["STATIC_SCALE"])
    with torch.no_grad():
        seg_map, height_field = tdp(volume.unsqueeze(dim=0))

    seg_map[seg_map == 0] = 4  # Assign the rest area as water area (as in City Sample)
    return seg_map.squeeze(dim=0), height_field.squeeze(dim=0), freeways


def get_footprint_bboxes(seg_map):
    building_instances = [
        i
        for i in np.unique(seg_map)
        if i >= CONSTANTS["BLD_INS_MIN_ID"] and i < CONSTANTS["CAR_INS_MIN_ID"]
    ]
    bboxes = {}
    for bi in tqdm(
        building_instances, desc="Generating Building Bounding Boxes", leave=False
    ):
        bboxes[bi] = cv2.boundingRect((seg_map == bi).astype(np.uint8))

    return bboxes


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
    coordinates[:, 2] = coordinates[:, 2] / scale + CONSTANTS["VOL_DEPTH_OFFSET"]  # z
    coordinates = (coordinates + 0.5).long()
    logging.debug("Volume Size (HxWxD): %s" % (volume.size(),))
    logging.debug(
        "X Min: %d, X Max: %d"
        % (torch.min(coordinates[:, 0]), torch.max(coordinates[:, 0]))
    )
    logging.debug(
        "Y Min: %d, X Max: %d"
        % (torch.min(coordinates[:, 1]), torch.max(coordinates[:, 1]))
    )
    logging.debug(
        "Z Min: %d, X Max: %d"
        % (torch.min(coordinates[:, 2]), torch.max(coordinates[:, 2]))
    )
    assert (coordinates[:, 0] < volume.size(1)).all()
    assert (coordinates[:, 1] < volume.size(0)).all()
    assert (coordinates[:, 2] < volume.size(2)).all()
    assert (coordinates[:, 2] >= 0).all()

    volume[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = values
    return volume


def get_volume_with_roof_1f(height_field, seg_map, freeways=None):
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
    if freeways is not None:
        volume = get_volume(
            freeways, scale=CONSTANTS["STATIC_SCALE"], volume=volume.squeeze(dim=0)
        )
    return volume.squeeze(dim=0)


def get_camera_poses(cam_pose, volume_size):
    # TODO: Consider the scale for cars
    cam_pose["tx"] = (float(cam_pose["tx"]) / 100 + volume_size[1]) / CONSTANTS[
        "STATIC_SCALE"
    ]
    cam_pose["ty"] = (float(cam_pose["ty"]) / 100 + volume_size[0]) / CONSTANTS[
        "STATIC_SCALE"
    ]
    cam_pose["tz"] = (float(cam_pose["tz"]) / 100) / CONSTANTS[
        "STATIC_SCALE"
    ] + CONSTANTS["CAM_DEPTH_OFFSET"]
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


def get_voxel_raycasting(cam_rig, cam_pose, volume):
    cp = get_camera_poses(cam_pose, volume.size())
    return get_ray_voxel_intersection(
        cam_rig, cp["cam_position"], cp["cam_look_at"], volume
    )


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


def get_ambiguous_seg_mask(voxel_id, est_seg_map):
    ins_seg_map = voxel_id.squeeze()[..., 0].copy()
    # NOTE: In ins_seg_map, 4n and 4n+1 denote building facade and roof, respectively.
    #       In est_seg_map, 7 and 8 denote building facade and roof, respectively.
    ins_seg_map[ins_seg_map >= CONSTANTS["CAR_INS_MIN_ID"]] = CONSTANTS["CAR_CLS_ID"]
    ins_seg_map[
        (ins_seg_map >= CONSTANTS["BLD_INS_MIN_ID"]) & (ins_seg_map % 4 == 0)
    ] = CONSTANTS["BLD_FACADE_CLS_ID"]
    ins_seg_map[
        (ins_seg_map >= CONSTANTS["BLD_INS_MIN_ID"]) & (ins_seg_map % 4 == 1)
    ] = CONSTANTS["BLD_ROOF_CLS_ID"]
    return ins_seg_map == np.array(est_seg_map.convert("P"))


def main(data_dir, seg_map_file_pattern, is_debug):
    cities = sorted(os.listdir(data_dir))
    for city in tqdm(cities):
        with open(os.path.join(data_dir, city, "VOLUME.pkl"), "rb") as fp:
            volume = pickle.load(fp)

        logging.debug(
            "X Min: %d, X Max: %d" % (np.min(volume[:, 0]), np.max(volume[:, 0]))
        )
        logging.debug(
            "Y Min: %d, Y Max: %d" % (np.min(volume[:, 1]), np.max(volume[:, 1]))
        )
        logging.debug(
            "Z Min: %d, Z Max: %d" % (np.min(volume[:, 2]), np.max(volume[:, 2]))
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
            os.path.join(data_dir, city, "SegLayout.png")
        )
        Image.fromarray(height_field.cpu().numpy()).save(
            os.path.join(data_dir, city, "HeightField.png")
        )
        with open(os.path.join(data_dir, city, "Freeway.pkl"), "wb") as fp:
            pickle.dump(freeways, fp)

        # Generate footprint bounding boxes
        footprint_bboxes = get_footprint_bboxes(seg_map.cpu().numpy())
        with open(os.path.join(data_dir, city, "Footprints.pkl"), "wb") as fp:
            pickle.dump(footprint_bboxes, fp)

        # Rebuild 3D volume with roof and 1f
        volume = get_volume_with_roof_1f(height_field, seg_map, freeways)

        # Generate raycasting results
        raycasting_dir = os.path.join(data_dir, city, "Raycasting")
        os.makedirs(raycasting_dir, exist_ok=True)
        with open(os.path.join(data_dir, city, "CameraRig.json")) as fp:
            cam_rig = json.load(fp)

        rows = []
        with open(os.path.join(data_dir, city, "CameraPoses.csv")) as fp:
            reader = csv.DictReader(fp)
            rows = [r for r in reader]

        for r in tqdm(rows):
            raycasting = get_voxel_raycasting(
                cam_rig["cameras"]["CameraComponent"], r, volume
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
                ).save(os.path.join(raycasting_dir, "%04d.png" % int(r["id"])))
            else:
                est_seg_map = Image.open(
                    os.path.join(
                        data_dir, city, seg_map_file_pattern % (city, int(r["id"]))
                    )
                )
                # Change the order of channels for efficiency
                raycasting["depth2"] = raycasting["depth2"].permute(1, 2, 0, 3, 4)
                raycasting = {k: v.cpu().numpy() for k, v in raycasting.items()}
                with open(
                    os.path.join(raycasting_dir, "%04d.pkl" % int(r["id"])), "wb"
                ) as ofp:
                    raycasting["mask"] = get_ambiguous_seg_mask(
                        raycasting["voxel_id"], est_seg_map
                    )
                    pickle.dump(raycasting, ofp)


if __name__ == "__main__":
    logging.config.dictConfig(
        {
            "disable_existing_loggers": True,
            "format": "[%(levelname)s] %(asctime)s %(message)s",
            "level": logging.DEBUG,
            "version": 1,
        }
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_HOME, "data"))
    parser.add_argument("--seg_map", default="SemanticImage/%sSequence.%04d.png")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.seg_map, args.debug)
