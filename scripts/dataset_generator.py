# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-21 20:34:18
# @Email:  root@haozhexie.com

import argparse
import csv
import cv2
import json
import logging
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
    "CITY_SAMPLE": {
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
    "GOOGLE_EARTH": {
        "NULL": 0,
        "ROAD": 1,
        "BLDG_FACADE": 2,
        "GREEN_LANDS": 3,
        "CONSTRUCTION": 4,
        "WATER": 5,
        "ZONE": 6,
        "BLDG_ROOF": 7,
    },
}


SCALES = {
    "CITY_SAMPLE": {
        "ROAD": 10,
        "FWY_DECK": 10,
        "FWY_PILLAR": 5,
        "FWY_BARRIER": 2,
        "CAR": 1,
        "WATER": 50,
        "SKY": 50,
        "ZONE": 10,
        "BLDG_FACADE": 3,
        "BLDG_ROOF": 3,
    },
    "GOOGLE_EARTH": {
        "ROAD": 10,
        "BLDG_FACADE": 2,
        "GREEN_LANDS": 10,
        "CONSTRUCTION": 2,
        "WATER": 50,
        "ZONE": 10,
    },
}

CONSTANTS = {
    "CITY_SAMPLE": {
        "SCALE": 20,  # 5x -> 1m (100 cm): 5 pixels
        "WATER_Z": -17.5,
        "MAP_SIZE": 24576,
        "IMAGE_WIDTH": 1920,
        "IMAGE_HEIGHT": 1080,
        "CAR_INS_MIN_ID": 5000,
    },
    "GOOGLE_EARTH": {
        "IMAGE_WIDTH": 960,
        "IMAGE_HEIGHT": 540,
    },
    "ROOF_INS_OFFSET": 1,
    "LOCAL_MAP_SIZE": 2048,
    "PATCH_SIZE": 5000,
    "BLDG_INS_MIN_ID": 100,
}


def get_projections(dataset, city_dir, osm_dir):
    if dataset == "CITY_SAMPLE":
        return _get_city_sample_projections(city_dir)
    elif dataset == "GOOGLE_EARTH":
        return _get_google_earth_projections(city_dir, osm_dir)
    else:
        raise Exception("Unknown dataset: %s" % (dataset))


def _get_city_sample_projections(city_dir):
    HOU_CLASSES = {
        "ROAD": 1,
        "FWY_DECK": 2,
        "FWY_PILLAR": 3,
        "FWY_BARRIER": 4,
        "ZONE": 5,
        # The following classes are not appeared in the Houdini export
        # But are needed by the _get_point_maps function.
        "NULL": 0,
        "BLDG_FACADE": 6,
        "BLDG_ROOF": 7,
    }
    HOU_SCALES = {
        "ROAD": 10,
        "FWY_DECK": 10,
        "FWY_PILLAR": 5,
        "FWY_BARRIER": 2,
        "CAR": 2,  # 1 -> 2 to make it more dense
        "WATER": 50,
        "SKY": 50,
        "ZONE": 10,
        "BLDG_FACADE": 5,
        "BLDG_ROOF": 5,
    }

    points_file_path = os.path.join(city_dir, "POINTS.pkl")
    if not os.path.exists(points_file_path):
        logging.warning("File not found in %s" % (points_file_path))
        return {}

    with open(points_file_path, "rb") as fp:
        points = pickle.load(fp)

    projections = _get_city_sample_projection(points, HOU_CLASSES, HOU_SCALES)
    projections["REST"] = _get_city_sample_water_areas(
        projections["REST"], HOU_SCALES["WATER"]
    )
    return projections


def _get_city_sample_projection(points, classes, scales):
    car_rows = points[:, 3] >= CONSTANTS["CITY_SAMPLE"]["CAR_INS_MIN_ID"]
    fwy_rows = np.isin(
        points[:, 3],
        [
            classes["FWY_DECK"],
            classes["FWY_PILLAR"],
            classes["FWY_BARRIER"],
        ],
    )
    rest_rows = ~np.logical_or(car_rows, fwy_rows)
    return {
        "CAR": _get_city_sample_points_projection(points[car_rows], classes, scales),
        "FWY": _get_city_sample_points_projection(points[fwy_rows], classes, scales),
        "REST": _get_city_sample_points_projection(points[rest_rows], classes, scales),
    }


def _get_city_sample_points_projection(points, classes, scales):
    # assert points.dtype == np.int16
    INVERSE_INDEX = {v: k for k, v in classes.items()}
    ins_map = np.zeros(
        (CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"], CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"]),
        dtype=points.dtype,
    )
    tpd_hf = -1 * np.ones(
        (CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"], CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"]),
        dtype=points.dtype,
    )
    btu_hf = np.iinfo(points.dtype).max * np.ones(
        (CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"], CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"]),
        dtype=points.dtype,
    )
    for p in tqdm(points, leave=False):
        x, y, z, c_id = p
        if z < 0:
            continue

        c_name = INVERSE_INDEX[c_id] if c_id in INVERSE_INDEX else None
        if c_name is None:
            if c_id < CONSTANTS["CITY_SAMPLE"]["CAR_INS_MIN_ID"]:
                # No building roof instance ID in the Houdini export
                # assert c_id % 4 == 0, c_id
                c_name = "BLDG_FACADE"
            else:
                c_name = "CAR"

        s = scales[c_name]
        x += CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"] // 2
        y += CONSTANTS["CITY_SAMPLE"]["MAP_SIZE"] // 2

        # NOTE: The pts_map is generated by the _get_point_maps function.
        # pts_map[y, x] = True
        if tpd_hf[y, x] < z:
            tpd_hf[y : y + s, x : x + s] = z
            ins_map[y : y + s, x : x + s] = (
                CLASSES["CITY_SAMPLE"][c_name]
                if c_name not in ["BLDG_FACADE", "BLDG_ROOF", "CAR"]
                else c_id
            )
        if btu_hf[y, x] > z:
            btu_hf[y : y + s, x : x + s] = z

    seg_map = _get_city_sample_seg_map(ins_map)
    return {
        "PTS": _get_point_maps(seg_map, CLASSES["CITY_SAMPLE"], SCALES["CITY_SAMPLE"]),
        "INS": ins_map,
        "SEG": seg_map,
        "TD_HF": tpd_hf,
        "BU_HF": btu_hf,
    }


def _get_point_maps(seg_map, classes, scales):
    inverted_index = {v: k for k, v in classes.items()}
    pts_map = np.zeros(seg_map.shape, dtype=bool)
    classes = np.unique(seg_map)
    for c in classes:
        cls_name = inverted_index[c]
        if cls_name == "NULL":
            continue

        mask = seg_map == c
        pt_map = _get_point_map(seg_map.shape, scales[cls_name])
        pt_map[~mask] = False
        pts_map += pt_map

    return pts_map


def _get_point_map(map_size, stride):
    pts_map = np.zeros(map_size, dtype=bool)
    ys = np.arange(0, map_size[0], stride)
    xs = np.arange(0, map_size[1], stride)
    coords = np.stack(np.meshgrid(ys, xs), axis=-1).reshape(-1, 2)
    pts_map[coords[:, 0], coords[:, 1]] = True
    return pts_map


def get_seg_map_from_ins_map(dataset, ins_map):
    if dataset == "CITY_SAMPLE":
        return _get_city_sample_seg_map(ins_map)
    elif dataset == "GOOGLE_EARTH":
        return _get_google_earth_seg_map(ins_map)
    else:
        raise Exception("Unknown dataset: %s" % (dataset))


def _get_city_sample_seg_map(ins_map):
    ins_map = ins_map.copy()
    ins_map[ins_map >= CONSTANTS["CITY_SAMPLE"]["CAR_INS_MIN_ID"]] = CLASSES[
        "CITY_SAMPLE"
    ]["CAR"]
    ins_map[
        np.where((ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]) & (ins_map % 2))
    ] = CLASSES["CITY_SAMPLE"]["BLDG_ROOF"]
    ins_map[ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]] = CLASSES["CITY_SAMPLE"][
        "BLDG_FACADE"
    ]
    return ins_map


def _get_google_earth_seg_map(ins_map):
    raise NotImplementedError


def _get_city_sample_water_areas(projection, scale):
    # The rest areas are assigned as the water areas, which is aligned with CitySample.
    water_area = projection["INS"] == CLASSES["CITY_SAMPLE"]["NULL"]
    projection["INS"][water_area] = CLASSES["CITY_SAMPLE"]["WATER"]
    projection["SEG"][water_area] = CLASSES["CITY_SAMPLE"]["WATER"]
    projection["TD_HF"][water_area] = 0
    projection["BU_HF"][water_area] = 0

    wa_pts_map = _get_point_map(projection["PTS"].shape, scale)
    wa_pts_map[~water_area] = False
    projection["PTS"] += wa_pts_map
    return projection


@utils.helpers.static_vars(osm={})
def _get_google_earth_projections(city_dir, osm_dir):
    ZOOM_LEVEL = 18

    city_name = "-".join(os.path.basename(city_dir).split("-")[:2])
    # Cache the full height field and semantic map
    if city_name in _get_google_earth_projections.osm:
        (
            td_hf,
            bu_hf,
            seg_map,
            ins_map,
            pts_map,
            osm_metadata,
        ) = _get_google_earth_projections.osm[city_name]
    else:
        td_hf, seg_map, ins_map, osm_metadata = _get_osm_data(
            osm_dir, city_name
        )
        bu_hf = np.zeros_like(td_hf)
        pts_map = _get_point_maps(
            seg_map, CLASSES["GOOGLE_EARTH"], SCALES["GOOGLE_EARTH"]
        )
        _get_google_earth_projections.osm[city_name] = (
            td_hf,
            bu_hf,
            seg_map,
            ins_map,
            pts_map,
            osm_metadata,
        )

    # Determine the local projection area
    ge_project_name = os.path.basename(city_dir)
    ge_project_file = os.path.join(city_dir, "%s.esp" % ge_project_name)
    ge_metadata_file = os.path.join(city_dir, "metadata.json")
    with open(ge_project_file) as f:
        ge_project_settings = json.load(f)
    with open(ge_metadata_file) as f:
        ge_metadata = json.load(f)

    cam_target = _get_google_earth_camera_target(ge_project_settings, ge_metadata)
    cx, cy = _lnglat2xy(
        cam_target["longitude"],
        cam_target["latitude"],
        osm_metadata["resolution"],
        ZOOM_LEVEL,
        dtype=float,
    )
    cx -= osm_metadata["bounds"]["xmin"]
    cy -= osm_metadata["bounds"]["ymin"]
    x_min = int(cx - CONSTANTS["LOCAL_MAP_SIZE"] // 2)
    x_max = int(cx + CONSTANTS["LOCAL_MAP_SIZE"] // 2)
    y_min = int(cy - CONSTANTS["LOCAL_MAP_SIZE"] // 2)
    y_max = int(cy + CONSTANTS["LOCAL_MAP_SIZE"] // 2)

    # Reorganze the instance ID of buildings
    reorg_ins_map = ins_map[y_min:y_max, x_min:x_max].copy()
    reorg_instances = np.unique(reorg_ins_map)
    n_bldg = CONSTANTS["BLDG_INS_MIN_ID"]
    for ri in reorg_instances:
        if ri < CONSTANTS["BLDG_INS_MIN_ID"]:
            continue
        reorg_ins_map[reorg_ins_map == ri] = n_bldg
        n_bldg += 2

    return {
        "REST": {
            "PTS": pts_map[y_min:y_max, x_min:x_max],
            "INS": reorg_ins_map,
            "SEG": seg_map[y_min:y_max, x_min:x_max],
            "TD_HF": td_hf[y_min:y_max, x_min:x_max],
            "BU_HF": bu_hf[y_min:y_max, x_min:x_max],
        }
    }


def _get_osm_data(osm_dir, city_name):
    osm_dir = os.path.join(osm_dir, city_name)

    height_field = np.array(Image.open(os.path.join(osm_dir, "hf.png")))
    semantic_map = np.array(Image.open(os.path.join(osm_dir, "seg.png")).convert("P"))
    with open(os.path.join(osm_dir, "metadata.json")) as f:
        metadata = json.load(f)

    instance_map, _ = _get_google_earth_instance_map(semantic_map.copy())
    return height_field, semantic_map, instance_map, metadata


# https://github.com/hzxie/CityDreamer/blob/master/scripts/dataset_generator.py#L393
def _get_google_earth_instance_map(seg_map):
    _, labels, centers, _ = cv2.connectedComponentsWithStats(
        (seg_map == CLASSES["GOOGLE_EARTH"]["BLDG_FACADE"]).astype(np.uint8),
        connectivity=4,
    )
    # Remove non-building instance masks
    labels[seg_map != CLASSES["GOOGLE_EARTH"]["BLDG_FACADE"]] = 0
    # Building instance mask
    building_mask = labels != 0

    # Make building instance IDs are even numbers and start from 10
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
    labels = (labels + CONSTANTS["BLDG_INS_MIN_ID"]) * 2

    seg_map[seg_map == CLASSES["GOOGLE_EARTH"]["BLDG_FACADE"]] = 0
    seg_map = seg_map * (1 - building_mask) + labels * building_mask
    assert np.max(labels) < 2147483648
    return seg_map.astype(np.int32), centers[:, :4]


# https://github.com/hzxie/CityDreamer/blob/master/scripts/dataset_generator.py#L292
def _get_google_earth_camera_target(project_settings, metadata):
    scene = project_settings["scenes"][0]["attributes"]
    camera_group = next(
        _attr["attributes"] for _attr in scene if _attr["type"] == "cameraGroup"
    )
    camera_taget_effect = next(
        _attr["attributes"]
        for _attr in camera_group
        if _attr["type"] == "cameraTargetEffect"
    )
    camera_target = next(
        _attr["attributes"] for _attr in camera_taget_effect if _attr["type"] == "poi"
    )
    return {
        "longitude": next(
            _attr["value"]["relative"]
            for _attr in camera_target
            if _attr["type"] == "longitudePOI"
        )
        * 360
        - 180,
        # NOTE: The conversion from latitudePOI to latitude is unclear.
        # Fixed with the collected metadata.
        "latitude": metadata["clat"],
    }


# https://github.com/hzxie/CityDreamer/blob/master/utils/osm_helper.py#L264
def _lnglat2xy(lng, lat, resolution, zoom_level, tile_size=256, dtype=int):
    # Ref: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    n = 2.0**zoom_level
    x = (lng + 180.0) / 360.0 * n * tile_size
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n * tile_size
    return (dtype(x * resolution), dtype(y * resolution))


def dump_projections(projections, output_dir, is_debug):
    os.makedirs(output_dir, exist_ok=True)
    for c, p in projections.items():
        for k, v in p.items():
            out_fpath = os.path.join(output_dir, "%s-%s.png" % (c, k))
            if is_debug:
                if k in ["SEG", "INS"]:
                    utils.helpers.get_seg_map(v).save(out_fpath)
                else:
                    Image.fromarray((v / np.max(v) * 255).astype(np.uint8)).save(
                        out_fpath
                    )
            elif k == "SEG":
                utils.helpers.get_seg_map(v).save(out_fpath)
            else:
                Image.fromarray(v.astype(np.int16)).save(out_fpath)


def load_projections(output_dir):
    CATEGORIES = ["CAR", "FWY", "REST"]
    MAP_NAMES = ["INS", "SEG", "TD_HF", "BU_HF", "PTS"]

    projections = {}
    for c in CATEGORIES:
        for m in MAP_NAMES:
            out_fpath = os.path.join(output_dir, "%s-%s.png" % (c, m))
            if not os.path.exists(out_fpath):
                continue
            if c not in projections:
                projections[c] = {}

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
    centers = {}
    for c, p in projections.items():
        instances = np.unique(p["INS"])
        # Append SKY to instances. Since SKY is not in the semantic map.
        instances = np.append(instances, CLASSES["CITY_SAMPLE"]["SKY"])

        for i in tqdm(instances, desc="Calculating centers for %s" % c):
            if i >= CONSTANTS["BLDG_INS_MIN_ID"]:
                ds_mask = p["INS"] == i
                contours, _ = cv2.findContours(
                    ds_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                contours = np.vstack(contours).reshape(-1, 2)
                min_x, max_x = np.min(contours[:, 0]), np.max(contours[:, 0])
                min_y, max_y = np.min(contours[:, 1]), np.max(contours[:, 1])
                max_z = np.max(p["TD_HF"][ds_mask]) + 1
            else:
                min_x, max_x = 0, p["TD_HF"].shape[1]
                min_y, max_y = 0, p["TD_HF"].shape[0]
                max_z = np.max(p["TD_HF"])
                # FWY_DECK and ROAD share the same semantic ID
                if i in centers:
                    max_z = max(max_z, centers[i][-1])

            centers[i] = np.array(
                [
                    (min_x + max_x) / 2,
                    (min_y + max_y) / 2,
                    (max_x - min_x),
                    (max_y - min_y),
                    max_z,
                ],
                dtype=np.float32,
            )
            # Fix the centers for BLDG_ROOF
            if (
                i >= CONSTANTS["BLDG_INS_MIN_ID"]
                and i < CONSTANTS["CITY_SAMPLE"]["CAR_INS_MIN_ID"]
            ):
                centers[i + 1] = centers[i]

    return centers


def get_camera_parameters(dataset, city_dir):
    if dataset == "CITY_SAMPLE":
        return get_city_sample_camera_parameters(city_dir)
    elif dataset == "GOOGLE_EARTH":
        return get_google_earth_camera_parameters(city_dir)
    else:
        raise Exception("Unknown dataset: %s" % (dataset))


def get_city_sample_camera_parameters(city_dir):
    with open(os.path.join(city_dir, "CameraRig.json")) as fp:
        cam_rig = json.load(fp)
        cam_rig = cam_rig["cameras"]["CameraComponent"]
        # render images with different resolution
        cam_rig["intrinsics"][0] /= 1920 / CONSTANTS["CITY_SAMPLE"]["IMAGE_WIDTH"]
        cam_rig["intrinsics"][4] /= 1080 / CONSTANTS["CITY_SAMPLE"]["IMAGE_HEIGHT"]
        cam_rig["intrinsics"][2] = CONSTANTS["CITY_SAMPLE"]["IMAGE_WIDTH"] // 2
        cam_rig["intrinsics"][5] = CONSTANTS["CITY_SAMPLE"]["IMAGE_HEIGHT"] // 2
        cam_rig["sensor_size"] = [
            CONSTANTS["CITY_SAMPLE"]["IMAGE_WIDTH"],
            CONSTANTS["CITY_SAMPLE"]["IMAGE_HEIGHT"],
        ]

    camera_poses = []
    with open(os.path.join(city_dir, "CameraPoses.csv")) as fp:
        reader = csv.DictReader(fp)
        camera_poses = [r for r in reader]

    return cam_rig, camera_poses


def get_google_earth_camera_parameters(city_dir):
    pass


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


def get_local_projections(projections, local_cords):
    MAPS = [
        {"name": "SEG", "dtype": np.uint8, "interpolation": cv2.INTER_NEAREST},
        {"name": "TD_HF", "dtype": np.float32, "interpolation": cv2.INTER_AREA},
    ]

    local_projections = {m["name"]: projections[m["name"]].copy() for m in MAPS}
    points = np.array([local_cords[1], local_cords[2], local_cords[3], local_cords[4]])
    # Crop the image
    x_min, x_max, y_min, y_max = _get_crop(points)
    for m in MAPS:
        m_name = m["name"]
        m_type = m["dtype"]
        local_projections[m_name] = local_projections[m_name][
            y_min:y_max, x_min:x_max
        ].astype(m_type)

    points -= [x_min, y_min]
    # Rotate the image
    M, width, height = _get_rotation(points)
    for m in MAPS:
        m_name = m["name"]
        m_intp = m["interpolation"]
        local_projections[m_name] = cv2.resize(
            cv2.warpPerspective(local_projections[m_name], M, (width, height)),
            (CONSTANTS["LOCAL_MAP_SIZE"], CONSTANTS["LOCAL_MAP_SIZE"]),
            interpolation=m_intp,
        )

    local_projections["tlp"] = np.array([x_min, y_min])
    local_projections["affmat"] = M
    return local_projections


def _get_crop(points):
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()
    return x_min, x_max, y_min, y_max


def _get_rotation(points):
    width = np.linalg.norm(points[0] - points[1])
    # assert abs(np.linalg.norm(points[2] - points[3]) - width) < 1
    height = np.linalg.norm(points[1] - points[2])
    # assert abs(np.linalg.norm(points[0] - points[3]) - height) < 1
    src_pts = np.array(points, dtype=np.float32)
    dst_pts = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_pts, dst_pts), int(width), int(height)


def get_points_from_projections(
    projections, classes, scales, seg_ins_map, local_cords=None
):
    # XYZ, Scale, Instance ID
    points = np.empty((0, 5), dtype=np.int16)
    for c, p in projections.items():
        # Ignore bottom points for objects in the rest maps due to invisibility.
        _points = _get_points_from_projection(
            p, classes, scales, seg_ins_map, local_cords, c != "REST"
        )
        if _points is not None:
            points = np.concatenate((points, _points), axis=0)
            logging.debug(
                "Category: %s: #Points: %d, Min/Max Value: (%d, %d)"
                % (c, len(_points), np.min(_points), np.max(_points))
            )
        # Move the water plane to -3.5m, which is aligned with CitySample.
        if c == "REST":
            points[:, 2][points[:, 4] == CLASSES["CITY_SAMPLE"]["WATER"]] = CONSTANTS[
                "CITY_SAMPLE"
            ]["WATER_Z"]

    logging.debug("#Points: %d" % (len(points)))
    return points


def _get_points_from_projection(
    projection, classes, scales, seg_ins_map, local_cords=None, include_btm_pts=True
):
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

    assert np.max(_projection["INS"]) < 32768
    points = footprint_extruder.get_points_from_projection(
        include_btm_pts,
        {v: k for k, v in classes.items()},
        scales,
        seg_ins_map,
        np.ascontiguousarray(_projection["INS"].astype(np.int16)),
        np.ascontiguousarray(_projection["TD_HF"].astype(np.int16)),
        np.ascontiguousarray(_projection["BU_HF"].astype(np.int16)),
        np.ascontiguousarray(_projection["PTS"].astype(bool)),
    )
    if points is not None and local_cords is not None:
        # Recover the XY coordinates before cropping
        points[:, 0] += min_x
        points[:, 1] += min_y

    return points.astype(np.int16) if points is not None else None


def get_sky_points(far_plane, cam_z, cam_fov_y, scale, class_id):
    points = []
    # Determine the border of sky
    sky_height = CONSTANTS["PATCH_SIZE"] * math.tan(cam_fov_y)
    z_min = math.floor(max(0, cam_z - sky_height))
    z_max = math.ceil(cam_z + sky_height)
    dist = np.linalg.norm(far_plane[0] - far_plane[1])
    n_plane_segs = math.ceil(dist / scale)
    slope = (far_plane[1] - far_plane[0]) / dist
    # Generate sky points
    for i in range(n_plane_segs):
        x = far_plane[0, 0] + i * scale * slope[0]
        y = far_plane[0, 1] + i * scale * slope[1]
        for z in range(z_min, z_max + 1, scale):
            points.append([x, y, z, scale, class_id])

    logging.debug("#Sky points: %d" % (len(points)))
    return np.array(points, dtype=np.int16)


def _get_localized_pt_cords(points, offsets):
    points[:, 0] -= offsets[0]
    points[:, 1] -= offsets[1]
    points[:, 2] -= offsets[2] - 1
    return points


def _get_volume(points, scales):
    x_min, x_max = torch.min(points[:, 0]).item(), torch.max(points[:, 0]).item()
    y_min, y_max = torch.min(points[:, 1]).item(), torch.max(points[:, 1]).item()
    z_min, z_max = torch.min(points[:, 2]).item(), torch.max(points[:, 2]).item()
    offsets = np.array([x_min, y_min, z_min], dtype=np.int16)
    # Normalize points coordinates to local coordinate system
    points = _get_localized_pt_cords(points, offsets)
    # Generate an empty 3D volume
    w, h, d = x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 2
    # Generate point IDs
    pt_ids = torch.arange(
        points.shape[0], dtype=torch.int32, device=points.device
    ).unsqueeze(dim=1)
    volume = extensions.voxlib.points_to_volume(
        points.contiguous(), pt_ids, scales, h, w, d
    )
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

    return voxel_id.squeeze()


def get_visible_points(points, scales, cam_rig, cam_pos, cam_quat):
    # NOTE: Each point is assigned with a unique ID. The values in the rendered map
    # denotes the visibility of the points. The values are the same as the point IDs.
    instances = torch.from_numpy(points[:, 4]).cuda()
    points = torch.from_numpy(points[:, [0, 1, 2]]).cuda()
    scales = (
        torch.from_numpy(scales).cuda()
        if isinstance(scales, np.ndarray)
        else scales.cuda()
    )
    # Scale the volume by 0.2 to reduce the memory usage
    cam_pos = cam_pos.copy() / 5.0
    scales = torch.ceil(scales / 5.0).short()
    points = torch.floor(points / 5.0).short()
    # Generate 3D volume
    volume, offsets = _get_volume(points, scales)
    # Ray-voxel intersection
    cam_pos -= offsets
    cam_look_at = utils.helpers.get_camera_look_at(cam_pos, cam_quat)
    vp_map = _get_ray_voxel_intersection(cam_rig, cam_pos, cam_look_at, volume)
    # Image.fromarray(
    #     utils.helpers.get_ins_seg_map.r_palatte[vp_map.cpu().numpy() % 16384],
    # ).save("output/test.jpg")

    # Generate the instance segmentation map as a side product
    ins_map = instances[vp_map]
    # Image.fromarray(
    #     utils.helpers.get_ins_seg_map.r_palatte[ins_map.cpu().numpy()],
    # ).save("output/test.jpg")
    return vp_map.cpu().numpy(), ins_map.cpu().numpy()


def main(dataset, data_dir, osm_dir, seg_map_file_pattern, gpus, is_debug):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    assert dataset in ["GOOGLE_EARTH", "CITY_SAMPLE"], "Unknown dataset: %s" % dataset

    cities = sorted(os.listdir(data_dir))
    for city in tqdm(cities):
        logging.info("Generating point projections...")
        city_dir = os.path.join(data_dir, city)
        projections = get_projections(dataset, city_dir, osm_dir)

        logging.info("Saving projections...")
        proj_dir = os.path.join(city_dir, "Projection")
        dump_projections(projections, proj_dir, is_debug)

        # # Debug: Load projection caches without computing
        with open("/tmp/projections.pkl", "wb") as fp:
            pickle.dump(projections, fp)
        # logging.info("loading projections...")
        # proj_dir = os.path.join(city_dir, "Projection")
        # projections = load_projections(proj_dir)
        # with open("/tmp/projections.pkl", "rb") as fp:
        #     projections = pickle.load(fp)

        logging.info("Calculate the XY center for instances...")
        if os.path.exists(os.path.join(city_dir, "CENTERS.pkl")):
            with open(os.path.join(city_dir, "CENTERS.pkl"), "rb") as fp:
                centers = pickle.load(fp)
        else:
            centers = get_centers_from_projections(projections)
            with open(os.path.join(city_dir, "CENTERS.pkl"), "wb") as fp:
                pickle.dump(centers, fp)

        # Construct the relationship between instance ID and semantic ID
        seg_ins_map = {
            # BLDG
            "ROOF_INS_OFFSET": CONSTANTS["ROOF_INS_OFFSET"],
            "BLDG_INS_MIN_ID": CONSTANTS["BLDG_INS_MIN_ID"],
            "BLDG_FACADE_SEMANTIC_ID": CLASSES[dataset]["BLDG_FACADE"],
            "BLDG_ROOF_SEMANTIC_ID": CLASSES[dataset]["BLDG_ROOF"],
            # CAR
            "CAR_INS_MIN_ID": CONSTANTS[dataset]["CAR_INS_MIN_ID"]
            if "CAR_INS_MIN_ID" in CONSTANTS[dataset]
            else 32767,
            "CAR_SEMANTIC_ID": CLASSES[dataset]["CAR"]
            if "CAR" in CLASSES[dataset]
            else 32767,
        }
        # Debug: Generate all initial points (casues OOM in rasterization)
        logging.info("Generate the initial points for the whole city...")
        # points[:, M] -> 0:3: XYZ, 3: Scale, 4: Instance ID
        points = get_points_from_projections(
            projections, CLASSES[dataset], SCALES[dataset], seg_ins_map
        )

        # # Debug: Point Cloud Visualization
        logging.info("Saving the generated point cloud...")
        xyz = points[:, :3]
        rgbs = utils.helpers.get_ins_colors(points[:, 4])
        utils.helpers.dump_ptcloud_ply("/tmp/points.ply", xyz, rgbs)

        # Load camera parameters
        cam_rig, cam_poses = get_camera_parameters(dataset, city_dir)

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
        os.makedirs(city_points_dir, exist_ok=True)
        for r in tqdm(cam_poses, desc="Rendering Gaussian Points"):
            cam_quat = np.array([r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float32)
            cam_pos = (
                np.array([r["tx"], r["ty"], r["tz"]], dtype=np.float32)
                / CONSTANTS["SCALE"]
            )
            cam_pos[0] += CONSTANTS["MAP_SIZE"] // 2
            cam_pos[1] += CONSTANTS["MAP_SIZE"] // 2
            cam_look_at = utils.helpers.get_camera_look_at(cam_pos, cam_quat)
            logging.debug("Current Camera: %s, Look at: %s" % (cam_pos, cam_look_at))
            view_frustum_cords = get_view_frustum_cords(
                cam_pos,
                cam_look_at,
                CONSTANTS["PATCH_SIZE"],
                # TODO: 1.5 -> 2.0. But 2.0 causes incomplete rendering.
                fov_x / 1.5,
            )
            local_projections = get_local_projections(
                projections["REST"], view_frustum_cords
            )
            points = get_points_from_projections(projections, view_frustum_cords)
            sky_points = get_sky_points(view_frustum_cords[1:3], cam_pos[2], fov_y / 2)
            points = np.concatenate((points, sky_points), axis=0)
            scales = utils.helpers.get_point_scales(points[:, [3]], points[:, [4]])

            # Generate the instance segmentation map as a side product.
            vp_map, ins_map = get_visible_points(
                points, scales, cam_rig, cam_pos, cam_quat
            )
            vp_idx = np.sort(np.unique(vp_map))
            # Remove the points that are not visible in the current view.
            points = points[vp_idx]
            logging.debug("%d points in frame %d." % (len(points), int(r["id"])))
            # Re-generate the visible points map in the newly indexed points
            vp_map = np.searchsorted(vp_idx, vp_map)

            # Image.fromarray(ins_map.astype(np.uint16)).save(
            #     os.path.join(city_dir, "InstanceImage", "%04d.png" % int(r["id"]))
            # )
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
                        "prj": local_projections,
                        "vpm": vp_map,
                        "msk": get_seg_map_from_ins_map(dataset, ins_map) == seg_map,
                        "pts": points,
                    },
                    fp,
                )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GOOGLE_EARTH")
    parser.add_argument(
        "--data_dir", default=os.path.join(PROJECT_HOME, "data", "google-earth")
    )
    parser.add_argument("--osm_dir", default=os.path.join(PROJECT_HOME, "data", "osm"))
    # parser.add_argument("--seg_map", default="SemanticImage/%sSequence.%04d.png")
    parser.add_argument("--seg_map", default="seg/%s_%02d.png")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.data_dir, args.osm_dir, args.seg_map, args.gpu, args.debug)
