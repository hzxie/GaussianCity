# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-26 09:20:41
# @Email:  root@haozhexie.com

import argparse
import csv
import cv2
import json
import logging
import logging.config
import math
import numpy as np
import os
import pickle
import sys
import torch

from PIL import Image
from tqdm import tqdm


PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import extensions.voxlib
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

# Should be aligned with Houdini Settings
SCALES = {
    "ROAD": 10,
    "FWY_DECK": 10,
    "FWY_PILLAR": 5,
    "FWY_BARRIER": 2,
    "CAR": 1,
    "WATER": 50,
    "SKY": 50,
    "ZONE": 10,
    "BLDG_FACADE": 5,
    "BLDG_ROOF": 5,
}

CONSTANTS = {
    "SCALE": 20,  # 5x -> 1m (100 cm): 5 pixels
    "WATER_Z": -17.5,
    "MAP_SIZE": 24576,
    "IMAGE_WIDTH": 1920,
    "IMAGE_HEIGHT": 1080,
    "PATCH_SIZE": 5000,
    "BLDG_INS_MIN_ID": 100,
    "CAR_INS_MIN_ID": 5000,
    "INS_MAP_MIN_PIXELS": 200,
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
    # assert points.dtype == np.int16
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
        if z < 0:
            continue

        c_name = INVERSE_INDEX[c_id] if c_id in INVERSE_INDEX else None
        if c_name is None:
            if c_id < CONSTANTS["CAR_INS_MIN_ID"]:
                # No building roof instance ID in the Houdini export
                # assert c_id % 4 == 0, c_id
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
                "Map[%s/%s] Min/Max Value: (%d, %d) Size: %s"
                % (
                    c,
                    m,
                    np.min(projections[c][m]),
                    np.max(projections[c][m]),
                    projections[c][m].shape,
                )
            )
    return projections


def get_centers_from_projections(projections):
    scale = 1
    centers = {}
    for c, p in projections.items():
        ds_seg = p["SEG"]
        if c == "CAR":
            scale = 1
        else:
            scale = 4
            ds_seg = cv2.resize(
                p["SEG"],
                dsize=(CONSTANTS["MAP_SIZE"] // scale, CONSTANTS["MAP_SIZE"] // scale),
                interpolation=cv2.INTER_NEAREST,
            )

        instances = np.unique(ds_seg)
        for i in tqdm(instances, leave=False, desc="Calculating centers for %s" % c):
            contours, _ = cv2.findContours(
                (ds_seg == i).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contours = np.vstack(contours).reshape(-1, 2)
            min_x, max_x = np.min(contours[:, 0]), np.max(contours[:, 0])
            min_y, max_y = np.min(contours[:, 1]), np.max(contours[:, 1])
            centers[i] = (
                np.array([(min_x + max_x) / 2, (min_y + max_y) / 2], dtype=np.int16)
                * scale
            )
    return centers


def get_view_frustum_cords(cam_pos, cam_look_at, patch_size, fov_rad):
    # cam_pos: (x1, y1); cam_look_at: (x2, y2)
    # This patch has four edges (E1, E2, E3, and E4) arranged clockwise.
    # (x1, y1) is on the center of E1. (x3, y3) is on the center of E3.
    x1, y1, x2, y2 = cam_pos[0], cam_pos[1], cam_look_at[0], cam_look_at[1]
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    x3, y3 = x1 + dx * patch_size, y1 + dy * patch_size
    # L1: the line connecting (x1, y1) and (x3, y3) -> y = kl1 * x + bl1
    # E3: the edge connecting (x3, y3) and (x4, y4), perpendicular to L1. ->
    # y = ke3 * x + be3, where ke3 = -1 / kl1
    kl1 = (y3 - y1) / (x3 - x1) if x3 != x1 else float("inf")
    ke3 = -1 / kl1 if kl1 != 0 else float("inf")
    be3 = y3 - ke3 * x3
    # L2: the line connecting (x1, y1) and (x4, y4); L3: the line connecting (x1, y1) and (x5, y5)
    # (x4, y4) and (x5, y5) are the two endpoints of E3, respectively.
    # L2 -> y = kl2 * x + bl2; L3 -> y = kl3 * x + bl3
    kl2 = math.tan(math.atan(kl1) + fov_rad)
    bl2 = y1 - kl2 * x1
    kl4 = math.tan(math.atan(kl1) - fov_rad)
    bl4 = y1 - kl4 * x1
    x4 = (bl2 - be3) / (ke3 - kl2)
    y4 = kl2 * x4 + bl2
    assert abs(y4 - ke3 * x4 - be3) < 1e-5
    x5 = (bl4 - be3) / (ke3 - kl4)
    y5 = kl4 * x5 + bl4
    assert abs(y5 - ke3 * x5 - be3) < 1e-5
    assert (x4 + x5) / 2 - x3 < 1e-5 and (y4 + y5) / 2 - y3 < 1e-5
    # (x6, y6) is the center of the rectangle
    x6, y6 = (x1 + x3) / 2, (y1 + y3) / 2
    # (x7, y7) and (x8, y8) are the two endpoints of E1, respectively.
    x7, y7 = 2 * x6 - x4, 2 * y6 - y4
    x8, y8 = 2 * x6 - x5, 2 * y6 - y5
    assert (x7 + x8) / 2 - x1 < 1e-5 and (y7 + y8) / 2 - y1 < 1e-5
    return np.array([(x1, y1), (x4, y4), (x5, y5), (x7, y7), (x8, y8)], dtype=np.int16)


def get_points_from_projections(projections, local_cords=None):
    # XYZ, Scale, Instance ID
    points = np.empty((0, 5), dtype=np.int16)
    for c, p in projections.items():
        # Ignore bottom points for objects in the rest maps due to invisibility.
        _points = _get_points_from_projection(p, local_cords, c != "REST")
        if _points is not None:
            points = np.concatenate((points, _points), axis=0)
            logging.debug(
                "Category: %s: #Points: %d, Min/Max Value: (%d, %d)"
                % (c, len(_points), np.min(_points), np.max(_points))
            )
        # Move the water plane to -3.5m, which is aligned with CitySample.
        if c == "REST":
            points[:, 2][points[:, 4] == CLASSES["GAUSSIAN"]["WATER"]] = CONSTANTS[
                "WATER_Z"
            ]

    logging.debug("#Points: %d" % (len(points)))
    return points


def _get_points_from_projection(projection, local_cords=None, include_btm_pts=True):
    _projection = projection
    if local_cords is not None:
        # local_cords contains 5 points
        # The first three points denotes the triangle of view frustum projection
        # The last four points denotes the minimum rectangle of the view frustum projection
        min_x = math.floor(np.min(local_cords[:, 0]))
        max_x = math.ceil(np.max(local_cords[:, 0]))
        min_y = math.floor(np.min(local_cords[:, 1]))
        max_y = math.ceil(np.max(local_cords[:, 1]))

        _projection = {}
        for c, p in projection.items():
            # The smallest bounding box of the minimum rectangle
            _projection[c] = np.ascontiguousarray(p[min_y:max_y, min_x:max_x]).astype(
                np.int16
            )
            if c == "PTS":
                mask = np.zeros_like(_projection[c], dtype=np.int16)
                cv2.fillPoly(
                    mask,
                    [np.array(local_cords - np.array([min_x, min_y]), dtype=np.int32)],
                    1,
                )
                _projection[c] = _projection[c] * mask

    points = footprint_extruder.get_points_from_projection(
        include_btm_pts,
        {v: k for k, v in CLASSES["GAUSSIAN"].items()},
        SCALES,
        _projection["SEG"],
        _projection["TD_HF"],
        _projection["BU_HF"],
        _projection["PTS"].astype(bool),
    )
    if points is not None and local_cords is not None:
        # Recover the XY coordinates before cropping
        points[:, 0] += min_x
        points[:, 1] += min_y

    return points.astype(np.int16) if points is not None else None


def get_scales(points):
    classes = points[:, 4]
    scales = np.ones((points.shape[0], 3), dtype=np.float32) * points[:, [3]]
    # Set the z-scale = 1 for roads, zones, and waters
    scales[:, 2][
        np.isin(
            classes,
            [
                CLASSES["GAUSSIAN"]["ROAD"],
                CLASSES["GAUSSIAN"]["WATER"],
                CLASSES["GAUSSIAN"]["ZONE"],
            ],
        )
    ] = 1
    return scales


def get_sky_points(far_plane, cam_z, cam_fov_y):
    points = []
    # Determine the border of sky
    sky_height = CONSTANTS["PATCH_SIZE"] * math.tan(cam_fov_y)
    z_min = math.floor(max(0, cam_z - sky_height))
    z_max = math.ceil(cam_z + sky_height)
    dist = np.linalg.norm(far_plane[0] - far_plane[1])
    n_plane_segs = math.ceil(dist / SCALES["SKY"])
    slope = (far_plane[1] - far_plane[0]) / dist
    # Generate sky points
    for i in range(n_plane_segs):
        x = far_plane[0, 0] + i * SCALES["SKY"] * slope[0]
        y = far_plane[0, 1] + i * SCALES["SKY"] * slope[1]
        for z in range(z_min, z_max + 1, SCALES["SKY"]):
            points.append([x, y, z, SCALES["SKY"], CLASSES["GAUSSIAN"]["SKY"]])

    logging.debug("#Sky points: %d" % (len(points)))
    return np.array(points, dtype=np.int16)


def _get_volume(points, scales):
    x_min, x_max = torch.min(points[:, 0]).item(), torch.max(points[:, 0]).item()
    y_min, y_max = torch.min(points[:, 1]).item(), torch.max(points[:, 1]).item()
    z_min, z_max = torch.min(points[:, 2]).item(), torch.max(points[:, 2]).item()
    offsets = np.array([x_min, y_min, z_min], dtype=np.int16)
    # Normalize points coordinates to local coordinate system
    points[:, 0] -= x_min
    points[:, 1] -= y_min
    # Make sure the minimum height is 1 because z = 0 indicates the height is 1.
    points[:, 2] -= z_min - 1
    # Generate an empty 3D volume
    w, h, d = x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 2
    # Naive Python Implementation (runtime ~ 5min)
    # volume = torch.zeros((h, w, d), dtype=torch.int16, device=points.device)
    # for i in tqdm(range(points.shape[0]), leave=False, desc="Generating 3D volume"):
    #     x, y, z, c = points[i]
    #     sx, sy, sz = scales[i]
    #     volume[y:y+sy, x:x+sx, z:z+sz] = c
    # CUDA Implementation
    volume = extensions.voxlib.points_to_volume(points.contiguous(), scales, h, w, d)
    return volume, offsets


def _get_ray_voxel_intersection(cam_rig, cam_position, cam_look_at, volume):
    N_MAX_SAMPLES = 1
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
    voxel_id, _, _ = extensions.voxlib.ray_voxel_intersection_perspective(
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
    # Manually release the memory to avoid OOM
    del volume
    torch.cuda.empty_cache()
    # No needed to map NULL voxels to SKY. Because the sky points are already generated.
    # voxel_id[voxel_id == 0] = CLASSES["GAUSSIAN"]["SKY"]
    return voxel_id.squeeze().cpu().numpy()


def get_ins_seg_map(points, scales, cam_rig, cam_pos, cam_quat):
    # points[:, 3] denotes the scale, which is duplicated with "scales"
    points = torch.from_numpy(points[:, [0, 1, 2, 4]]).cuda()
    scales = torch.from_numpy(scales).cuda()
    # Scale the volume by 0.2 to reduce the memory usage
    cam_pos = cam_pos.copy() / 5.0
    scales = torch.ceil(scales / 5.0).short()
    points[:, :3] = torch.floor(points[:, :3] / 5.0).short()
    # Generate 3D volume
    volume, offsets = _get_volume(points, scales)
    cam_pos -= offsets

    cam_look_at = utils.helpers.get_camera_look_at(cam_pos, cam_quat)
    return _get_ray_voxel_intersection(cam_rig, cam_pos, cam_look_at, volume)


def _get_seg_map_from_ins_map(ins_map):
    ins_map = ins_map.copy()
    ins_map[ins_map >= CONSTANTS["CAR_INS_MIN_ID"]] = CLASSES["GAUSSIAN"]["CAR"]
    ins_map[
        np.where((ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]) & (ins_map % 2))
    ] = CLASSES["GAUSSIAN"]["BLDG_ROOF"]
    ins_map[ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]] = CLASSES["GAUSSIAN"][
        "BLDG_FACADE"
    ]
    return ins_map


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

        logging.info("Saving projections...")
        proj_dir = os.path.join(city_dir, "Projection")
        dump_projections(projections, proj_dir, is_debug)

        # # Debug: Load projection caches without computing
        # with open("/tmp/projections.pkl", "wb") as fp:
        #     pickle.dump(projections, fp)
        # logging.info("loading projections...")
        # proj_dir = os.path.join(city_dir, "Projection")
        # projections = load_projections(proj_dir)
        # with open("/tmp/projections.pkl", "rb") as fp:
        #     projections = pickle.load(fp)

        logging.info("Calculate the XY center for instances...")
        centers = get_centers_from_projections(projections)
        with open(os.path.join(city_dir, "CENTERS.pkl"), "wb") as fp:
            pickle.dump(centers, fp)

        # # Debug: Generate all initial points (casues OOM in rasterization)
        # logging.info("Generate the initial points for the whole city...")
        # # points[:, M] -> 0:3: XYZ, 3: Scale, 4: Instance ID
        # points = get_points_from_projections(projections)

        # # Debug: Point Cloud Visualization
        # logging.info("Saving the generated point cloud...")
        # xyz = points[:, :3]
        # rgbs = utils.helpers.get_ins_colors(points[:, 4])
        # utils.helpers.dump_ptcloud_ply("/tmp/points.ply", xyz, rgbs)

        # Load camera parameters
        with open(os.path.join(city_dir, "CameraRig.json")) as fp:
            cam_rig = json.load(fp)
            cam_rig = cam_rig["cameras"]["CameraComponent"]
            # render images with different resolution
            cam_rig["intrinsics"][0] /= 1920 / CONSTANTS["IMAGE_WIDTH"]
            cam_rig["intrinsics"][4] /= 1080 / CONSTANTS["IMAGE_HEIGHT"]
            cam_rig["intrinsics"][2] = CONSTANTS["IMAGE_WIDTH"] // 2
            cam_rig["intrinsics"][5] = CONSTANTS["IMAGE_HEIGHT"] // 2
            cam_rig["sensor_size"] = [
                CONSTANTS["IMAGE_WIDTH"],
                CONSTANTS["IMAGE_HEIGHT"],
            ]

        rows = []
        with open(os.path.join(city_dir, "CameraPoses.csv")) as fp:
            reader = csv.DictReader(fp)
            rows = [r for r in reader]

        # Initialize the gaussian rasterizer
        fov_x = utils.helpers.intrinsic_to_fov(
            cam_rig["intrinsics"][0], cam_rig["sensor_size"][0]
        )
        fov_y = utils.helpers.intrinsic_to_fov(
            cam_rig["intrinsics"][4], cam_rig["sensor_size"][1]
        )
        logging.debug(
            "Camera FOV: (%.2f, %.2f) deg." % (np.rad2deg(fov_x), np.rad2deg(fov_y))
        )

        city_points_dir = os.path.join(city_dir, "Points")
        city_insseg_dir = os.path.join(city_dir, "InstanceImage")
        os.makedirs(city_points_dir, exist_ok=True)
        os.makedirs(city_insseg_dir, exist_ok=True)
        for r in tqdm(rows, desc="Rendering Gaussian Points"):
            cam_quat = np.array([r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float32)
            cam_pos = (
                np.array([r["tx"], r["ty"], r["tz"]], dtype=np.float32)
                / CONSTANTS["SCALE"]
            )
            cam_pos[0] += CONSTANTS["MAP_SIZE"] // 2
            cam_pos[1] += CONSTANTS["MAP_SIZE"] // 2
            cam_look_at = utils.helpers.get_camera_look_at(cam_pos, cam_quat)
            logging.debug("Current Camera: %s, Look at: %s" % (cam_pos, cam_look_at))
            local_cords = get_view_frustum_cords(
                cam_pos,
                cam_look_at,
                CONSTANTS["PATCH_SIZE"],
                # TODO: 1.5 -> 2.0. But 2.0 causes incomplete rendering.
                fov_x / 1.5,
            )
            points = get_points_from_projections(projections, local_cords)
            sky_points = get_sky_points(local_cords[1:3], cam_pos[2], fov_y / 2)
            points = np.concatenate((points, sky_points), axis=0)
            scales = get_scales(points)
            ins_map = get_ins_seg_map(points, scales, cam_rig, cam_pos, cam_quat)
            # # Debug: visualize the instance segmentation map
            # Image.fromarray(
            #     utils.helpers.get_ins_seg_map.r_palatte[ins_map],
            # ).save("output/render/%04d.jpg" % int(r["id"]))
            seg_map = np.array(
                Image.open(
                    os.path.join(city_dir, seg_map_file_pattern % (city, int(r["id"])))
                ).convert("P")
            )
            with open(
                os.path.join(city_points_dir, "%04d.pkl" % int(r["id"])), "wb"
            ) as fp:
                pickle.dump(
                    {
                        "ins": ins_map,
                        "msk": _get_seg_map_from_ins_map(ins_map) == seg_map,
                        "pts": points,
                    },
                    fp,
                )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_HOME, "data"))
    parser.add_argument("--seg_map", default="SemanticImage/%sSequence.%04d.png")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.seg_map, args.gpu, args.debug)
