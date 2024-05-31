# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-05-31 16:18:39
# @Email:  root@haozhexie.com

import argparse
import copy
import csv
import cv2
import json
import logging
import lxml.etree
import math
import numpy as np
import open3d
import os
import pickle
import scipy.spatial
import shutil
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
    # hzxie/city-dreamer/scripts/dataset_generator.py
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
    # autonomousvision/kitti360Scripts/kitti360scripts/helpers/labels.py
    "KITTI_360": {
        "NULL": 0,
        "ROAD": 1,
        "BLDG_FACADE": 2,
        "CAR": 3,
        "VEGETATION": 4,
        "SKY": 5,
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
        "BLDG_ROOF": 6,
    },
    "GOOGLE_EARTH": {
        "ROAD": 5,
        "BLDG_FACADE": 1,
        "BLDG_ROOF": 1,
        "GREEN_LANDS": 2,
        "CONSTRUCTION": 1,
        "WATER": 25,
        "ZONE": 5,
    },
    "KITTI_360": {
        "ROAD": 10,
        "BLDG_FACADE": 2,
        "CAR": 1,
        "VEGETATION": 2,
        "SKY": 25,
        "ZONE": 10,
        "BLDG_ROOF": 2,
    },
}

CONSTANTS = {
    "CITY_SAMPLE": {
        "SCALE": 20,  # 5x -> 1m (100 cm): 5 pixels
        "WATER_Z": -17.5,
        "MAP_SIZE": 24576,
        "PATCH_SIZE": 5000,
        "PROJECTION_SIZE": 2048,
        "IMAGE_WIDTH": 1920,
        "IMAGE_HEIGHT": 1080,
        "CAR_INS_MIN_ID": 5000,
        "SEG_MAP_PATTERN": "SemanticImage/%sSequence.%04d.png",
        "OUT_FILE_NAME_PATTERN": "%04d",
    },
    "GOOGLE_EARTH": {
        "SCALE": 1,
        "WATER_Z": 0,
        "MAP_SIZE": 2048,
        "PATCH_SIZE": 2048,
        "PROJECTION_SIZE": 2048,
        "IMAGE_WIDTH": 960,
        "IMAGE_HEIGHT": 540,
        "ZOOM_LEVEL": 18,
        "SEG_MAP_PATTERN": "seg/%s_%02d.png",
        "OUT_FILE_NAME_PATTERN": "%04d",
    },
    "KITTI_360": {
        "SCALE": 1,
        "CAR_SCALE": [0.5, 0.75, 0.75],
        "MAP_SIZE": 0,
        "VOXEL_SIZE": 0.1,
        "PATCH_SIZE": 1280,
        "PROJECTION_SIZE": 2048,
        "CAR_INS_MIN_ID": 10000,
        "SEG_MAP_PATTERN": "seg/%010d.png",
        "OUT_FILE_NAME_PATTERN": "%010d",
        "TREE_ASSETS_DIR": "/tmp/trees",
    },
    "ROOF_INS_OFFSET": 1,
    "BLDG_INS_MIN_ID": 100,
}


def reorganize_kitti_360(data_dir):
    # Reogranize the KITTI 360 by cities. Remove images without annotations.
    output_dir = os.path.join(data_dir, "processed")
    # Check whether the dataset has been reorganized
    if os.path.exists(os.path.join(output_dir, "DONE")):
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    cities = sorted(os.listdir(os.path.join(data_dir, "data_2d_raw")))
    for c in tqdm(cities, leave=False):
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)
        rgb_dir = os.path.join(data_dir, "data_2d_raw", c, "image_00", "data_rect")
        seg_dir = os.path.join(
            data_dir, "data_2d_semantics", "train", c, "image_00", "semantic"
        )
        # List images
        rgb_files = os.listdir(rgb_dir)
        seg_files = os.listdir(seg_dir)
        with open(os.path.join(data_dir, "data_poses", c, "cam0_to_world.txt")) as fp:
            poses = fp.read().splitlines()

        frames = [int(p.split(" ")[0]) for p in poses]
        # Reorganize files
        selected_poses = []
        for i, f in enumerate(tqdm(frames, leave=False)):
            f_name = "%010d.png" % f
            if f_name not in rgb_files or f_name not in seg_files:
                continue

            selected_poses.append(poses[i])
            os.makedirs(os.path.join(output_dir, c, "footage"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, c, "seg"), exist_ok=True)
            shutil.copy(
                os.path.join(rgb_dir, f_name),
                os.path.join(output_dir, c, "footage", f_name),
            )
            # shutil.copy(
            #     os.path.join(seg_dir, f_name),
            #     os.path.join(output_dir, c, "seg", f_name),
            # )
            # Rewrite the semantic maps with the defined classes
            # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
            seg_map = np.array(Image.open(os.path.join(seg_dir, f_name)).convert("P"))
            seg_map = _get_kitti_360_remapped_semantic_map(seg_map)
            utils.helpers.get_seg_map(seg_map).save(
                os.path.join(output_dir, c, "seg", f_name)
            )

        with open(os.path.join(output_dir, c, "cam0_to_world.txt"), "w") as fp:
            fp.write("\n".join(selected_poses))

    return output_dir


def _get_kitti_360_remapped_semantic_map(seg_map):
    MAPPER = {
        7: CLASSES["KITTI_360"]["ROAD"],  # road
        11: CLASSES["KITTI_360"]["BLDG_FACADE"],  # building
        26: CLASSES["KITTI_360"]["CAR"],  # car
        27: CLASSES["KITTI_360"]["CAR"],  # truck
        21: CLASSES["KITTI_360"]["VEGETATION"],  # vegetation
        23: CLASSES["KITTI_360"]["SKY"],  # sky
        6: CLASSES["KITTI_360"]["ZONE"],  # ground
        8: CLASSES["KITTI_360"]["ZONE"],  # sidewalk
    }
    return np.vectorize(lambda x: MAPPER.get(x, CLASSES["KITTI_360"]["NULL"]))(seg_map)


def get_projections(dataset, city_dir, osm_dir):
    if dataset == "CITY_SAMPLE":
        return _get_city_sample_projections(city_dir)
    elif dataset == "GOOGLE_EARTH":
        return _get_google_earth_projections(city_dir, osm_dir)
    elif dataset == "KITTI_360":
        return _get_kitti_360_projections(city_dir)
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
    return None, projections


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
        td_hf, seg_map, ins_map, osm_metadata = _get_osm_data(osm_dir, city_name)
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
        CONSTANTS["GOOGLE_EARTH"]["ZOOM_LEVEL"],
        dtype=float,
    )
    cx -= osm_metadata["bounds"]["xmin"]
    cy -= osm_metadata["bounds"]["ymin"]
    x_min = int(cx - CONSTANTS["GOOGLE_EARTH"]["MAP_SIZE"] // 2)
    x_max = int(cx + CONSTANTS["GOOGLE_EARTH"]["MAP_SIZE"] // 2)
    y_min = int(cy - CONSTANTS["GOOGLE_EARTH"]["MAP_SIZE"] // 2)
    y_max = int(cy + CONSTANTS["GOOGLE_EARTH"]["MAP_SIZE"] // 2)

    # Reorganze the instance ID of buildings
    reorg_ins_map = ins_map[y_min:y_max, x_min:x_max].copy()
    reorg_instances = np.unique(reorg_ins_map)
    n_bldg = CONSTANTS["BLDG_INS_MIN_ID"]
    for ri in reorg_instances:
        if ri < CONSTANTS["BLDG_INS_MIN_ID"]:
            continue
        reorg_ins_map[reorg_ins_map == ri] = n_bldg
        n_bldg += 2

    metadata = osm_metadata.copy()
    metadata["target"] = {"x": cx, "y": cy, "z": cam_target["altitude"]}
    return metadata, {
        "REST": {
            "PTS": pts_map[y_min:y_max, x_min:x_max],
            "INS": reorg_ins_map,
            "SEG": seg_map[y_min:y_max, x_min:x_max],
            # Consider the elevation of the local area
            "TD_HF": td_hf[y_min:y_max, x_min:x_max] + ge_metadata["elevation"],
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

    # Make building instance IDs are even numbers and start from 100
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k + 1.
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
        "altitude": next(
            _attr["value"]["relative"]
            for _attr in camera_target
            if _attr["type"] == "altitudePOI"
        )
        + 1,
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


def _get_kitti_360_projections(city_dir):
    bbox_3d_dir = os.path.join(
        city_dir, os.pardir, os.pardir, "data_3d_bboxes", "train_full"
    )
    city_name = os.path.basename(city_dir)
    xml_root = lxml.etree.parse(
        os.path.join(bbox_3d_dir, "%s.xml" % city_name)
    ).getroot()
    annotations = {}
    for c in tqdm(xml_root, leave=False):
        if c.find("transform") is None:
            continue

        fs, fe, bbox_3d = _get_kitti_360_3d_bbox_annotations(c)
        if bbox_3d is None:
            continue
        key = "%010d-%010d" % (fs, fe)
        if key not in annotations:
            annotations[key] = []
        annotations[key].append(bbox_3d)

    points = []
    instances = []
    for v in tqdm(annotations.values(), leave=False):
        p, i = _get_kitti_360_points(v)
        points.append(p)
        instances.append(i)

    points = np.concatenate(points, axis=0)
    instances = np.concatenate(instances, axis=0)
    # # Debug: Voxel Visualization
    # utils.helpers.dump_ptcloud_ply(
    #     "/tmp/points.ply",
    #     points,
    #     utils.helpers.get_ins_colors(instances),
    # )
    vegt_rows = instances == CLASSES["KITTI_360"]["VEGETATION"]
    rest_rows = ~vegt_rows
    vegt_metadata, vegt_projection = _get_kitti_360_projection(
        np.concatenate([points[vegt_rows], instances[vegt_rows, np.newaxis]], axis=1)
    )
    rest_metadata, rest_projection = _get_kitti_360_projection(
        np.concatenate([points[rest_rows], instances[rest_rows, np.newaxis]], axis=1)
    )

    # vegt_projection and rest_projection may have different sizes, which should be unified.
    metadata, projections = _get_kitti_360_merged_projections(
        {"VEGT": vegt_metadata, "REST": rest_metadata},
        {"VEGT": vegt_projection, "REST": rest_projection},
    )
    return metadata, projections


@utils.helpers.static_vars(
    car_counter=CONSTANTS["KITTI_360"]["CAR_INS_MIN_ID"],
    bldg_counter=CONSTANTS["BLDG_INS_MIN_ID"],
)
def _get_kitti_360_3d_bbox_annotations(xml_node):
    KITTI_CLASSES = {
        "road": CLASSES["KITTI_360"]["ROAD"],
        "driveway": CLASSES["KITTI_360"]["ROAD"],
        "building": CLASSES["KITTI_360"]["BLDG_FACADE"],
        "car": CLASSES["KITTI_360"]["CAR"],
        "truck": CLASSES["KITTI_360"]["CAR"],
        "vegetation": CLASSES["KITTI_360"]["VEGETATION"],
        "sky": CLASSES["KITTI_360"]["SKY"],
        "sidewalk": CLASSES["KITTI_360"]["ZONE"],
        "ground": CLASSES["KITTI_360"]["ZONE"],
    }
    frame_start = int(xml_node.find("start_frame").text)
    frame_end = int(xml_node.find("end_frame").text)
    is_dynamic = int(xml_node.find("dynamic").text) == 1

    bbox3d = None
    # category = xml_node.find('category').text
    label = xml_node.find("label").text
    if label in KITTI_CLASSES.keys() and not is_dynamic:
        instance_id = KITTI_CLASSES[label]
        if label in ["car"]:
            instance_id = _get_kitti_360_3d_bbox_annotations.car_counter
            _get_kitti_360_3d_bbox_annotations.car_counter += 1
        elif label in ["building"]:
            instance_id = _get_kitti_360_3d_bbox_annotations.bldg_counter
            _get_kitti_360_3d_bbox_annotations.bldg_counter += 2

        transform = _get_kitti_360_annotation_matrix(xml_node.find("transform"))
        vertices = _get_kitti_360_annotation_matrix(xml_node.find("vertices"))
        R = transform[:3, :3]
        t = transform[:3, 3]
        bbox3d = {
            "name": xml_node.tag,
            "instance": instance_id,
            "vertices": np.matmul(R, vertices.transpose()).transpose() + t,
            "faces": _get_kitti_360_annotation_matrix(xml_node.find("faces")).astype(
                np.int32
            ),
        }
        if label in ["building"]:
            # Manually generate roof vertices
            bbox3d = _get_kitti_360_bldg_roof(bbox3d)
        elif label in ["car"]:
            # The original annotations for cars are too large
            bbox3d = _get_scaled_kitti_360_car(
                bbox3d, CONSTANTS["KITTI_360"]["CAR_SCALE"]
            )
        elif label in ["vegetation"]:
            bbox3d = _get_kitti_360_trees(bbox3d)

    return frame_start, frame_end, bbox3d


def _get_kitti_360_annotation_matrix(xml_node):
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/annotation.py#L111-L123
    rows = int(xml_node.find("rows").text)
    cols = int(xml_node.find("cols").text)
    data = xml_node.find("data").text.split(" ")
    mat = []
    for d in data:
        d = d.replace("\n", "")
        if len(d) < 1:
            continue
        mat.append(float(d))

    mat = np.reshape(mat, [rows, cols])
    return mat


def _get_kitti_360_bldg_roof(bbox3d):
    vertices = bbox3d["vertices"]
    # Determine the bbox height
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    z_mid = z_min + (z_max - z_min) * 0.666
    pt1 = vertices[0, :2]
    pt2 = vertices[2, :2]
    pt3 = vertices[4, :2]
    pt4 = vertices[6, :2]
    d12 = np.linalg.norm(pt1 - pt2)
    d13 = np.linalg.norm(pt1 - pt3)
    d14 = np.linalg.norm(pt1 - pt4)
    # Determine the shorter side
    assert d14 > d13 and d14 > d12
    if d12 < d13:
        pt5 = (pt1 + pt2) / 2
        pt6 = (pt3 + pt4) / 2
    else:
        pt5 = (pt1 + pt3) / 2
        pt6 = (pt2 + pt4) / 2
    # Regenerate vertices
    bbox3d["vertices"] = np.array(
        [
            [pt1[0], pt1[1], z_min],
            [pt1[0], pt1[1], z_mid],
            [pt2[0], pt2[1], z_min],
            [pt2[0], pt2[1], z_mid],
            [pt3[0], pt3[1], z_min],
            [pt3[0], pt3[1], z_mid],
            [pt4[0], pt4[1], z_min],
            [pt4[0], pt4[1], z_mid],
            [pt5[0], pt5[1], z_max],
            [pt6[0], pt6[1], z_max],
        ]
    )
    # Regenerate faces
    bbox3d["faces"] = np.array(
        [
            [0, 2, 6],
            [0, 4, 6],
            [0, 1, 4],
            [1, 4, 5],
            [2, 3, 6],
            [3, 6, 7],
            [0, 1, 2],
            [1, 2, 3],
            [4, 5, 6],
            [5, 6, 7],
            [5, 8, 9],
            [5, 7, 9],
            [1, 3, 9],
            [1, 8, 9],
            [1, 5, 8],
            [3, 9, 7],
        ]
    )
    return bbox3d


def _get_scaled_kitti_360_car(bbox3d, scales):
    center = np.mean(bbox3d["vertices"], axis=0)
    # min_coords = np.min(bbox3d["vertices"], axis=0)
    # max_coords = np.max(bbox3d["vertices"], axis=0)
    # dimensions = max_coords - min_coords
    bbox3d["vertices"] = center + (bbox3d["vertices"] - center) * np.array(scales)
    return bbox3d


@utils.helpers.static_vars(trees=[])
def _get_kitti_360_trees(bbox3d):
    HEIGHT_THRESHOLD = 2.0
    QUAT_SCALE_FACTOR = 500
    TREE_INTERVAL = QUAT_SCALE_FACTOR * 2
    PROJECTION_SHRINK_KERNEL = TREE_INTERVAL // 8
    # Load the 3D tree models
    if not _get_kitti_360_trees.trees:
        tree_assets = [f for f in os.listdir(CONSTANTS["KITTI_360"]["TREE_ASSETS_DIR"])]
        for ta in tree_assets:
            asset = open3d.io.read_triangle_mesh(
                os.path.join(CONSTANTS["KITTI_360"]["TREE_ASSETS_DIR"], ta)
            )
            # Normalize the assets
            asset_center = (asset.get_max_bound() + asset.get_min_bound()) / 2
            asset_center[1] = asset.get_min_bound()[1]  # Move the center to the bottom
            asset_size = asset.get_max_bound() - asset.get_min_bound()
            asset_scale = np.min(asset_size)
            asset_vertices = np.asarray(asset.vertices)
            asset_vertices = (asset_vertices - asset_center) / asset_scale
            # Swap the Y and Z axes
            asset_vertices = asset_vertices[:, [0, 2, 1]]
            _get_kitti_360_trees.trees.append(
                {
                    "vertices": asset_vertices,
                    "faces": np.asarray(asset.triangles),
                }
            )

    faces = bbox3d["faces"]
    vertices = bbox3d["vertices"]
    # Check the bbox height
    min_z, max_z = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    if max_z - min_z < HEIGHT_THRESHOLD:
        return bbox3d

    # Determine the projection areas on the XY plane
    faces_2d = vertices[:, :2][faces]
    min_x, min_y = np.min(faces_2d[..., 0]), np.min(faces_2d[..., 1])
    tlp = np.array([min_x, min_y])
    faces_2d = ((faces_2d - tlp) * QUAT_SCALE_FACTOR).astype(np.int32)
    max_x, max_y = np.max(faces_2d[..., 0]), np.max(faces_2d[..., 1])
    mask = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
    for f in faces_2d:
        cv2.drawContours(mask, [f.astype(np.int32)], 0, 255, -1)
    # Shrink the projection area to avoid generate trees near the boundary
    mask = cv2.erode(
        mask, np.ones((TREE_INTERVAL, PROJECTION_SHRINK_KERNEL), np.uint8), iterations=1
    )
    # Determine the tree roots, starting from the top-left corner
    tree_roots = []
    for i in range(0, max_x, TREE_INTERVAL):
        for j in range(0, max_y, TREE_INTERVAL):
            if mask[j, i] != 0:
                tree_roots.append(np.array([i, j]) / QUAT_SCALE_FACTOR + tlp)
                cv2.circle(mask, (i, j), 10, 128, -1)
    # No trees should be generated, fallback to the original annotation
    if len(tree_roots) == 0:
        return bbox3d

    # Random replace 3D assets
    bbox3d["vertices"] = []
    bbox3d["faces"] = []
    for tr in tree_roots:
        tree = copy.deepcopy(np.random.choice(_get_kitti_360_trees.trees))
        vertices = tree["vertices"] + np.array([tr[0], tr[1], min_z])
        faces = tree["faces"] + len(bbox3d["vertices"])
        bbox3d["vertices"].append(vertices)
        bbox3d["faces"].append(faces)

    bbox3d["vertices"] = np.concatenate(bbox3d["vertices"], axis=0)
    bbox3d["faces"] = np.concatenate(bbox3d["faces"], axis=0)
    return bbox3d


def _get_kitti_360_points(bboxes):
    voxels = []
    instances = []
    for bbox in bboxes:
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(bbox["vertices"])
        mesh.triangles = open3d.utility.Vector3iVector(bbox["faces"])
        volume = open3d.geometry.VoxelGrid()
        volume = volume.create_from_triangle_mesh(
            mesh, voxel_size=CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
        )
        _voxels = np.array([v.grid_index for v in volume.get_voxels()])
        # Recover the absolute position of voxels
        min_x = np.min(bbox["vertices"][:, 0]) / CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
        min_y = np.min(bbox["vertices"][:, 1]) / CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
        min_z = np.min(bbox["vertices"][:, 2]) / CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
        if min_z < 0:
            logging.warning(
                "Ignore annotation %s due to incorrect annotations (%.2f, %.2f, %.2f)."
                % (bbox["name"], min_x, min_y, min_z)
            )
            continue

        _voxels[:, 0] += int(min_x)
        _voxels[:, 1] += int(min_y)
        _voxels[:, 2] += int(min_z)
        voxels.append(_voxels)
        instances.append([bbox["instance"]] * len(_voxels))

    return np.concatenate(voxels, axis=0), np.concatenate(instances, axis=0)


def _get_kitti_360_projection(points):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min = np.min(points[:, 2])

    ins_map = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.int16)
    tpd_hf = np.zeros_like(ins_map)
    btu_hf = np.iinfo(points.dtype).max * np.ones_like(ins_map)
    for p in tqdm(points, leave=False):
        x, y, z, i = p
        _x, _y, _z = x - x_min, y - y_min, z - z_min
        if i in [CLASSES["KITTI_360"]["ROAD"], CLASSES["KITTI_360"]["ZONE"]]:
            # The MAGIC NUMBER 3 makes the road and zone are better aligned with RGB images
            _z -= 3

        if tpd_hf[_y, _x] < _z:
            tpd_hf[_y, _x] = _z
            ins_map[_y, _x] = i
        if btu_hf[_y, _x] > _z:
            btu_hf[_y, _x] = _z

    seg_map = _get_kitti_360_seg_map(ins_map)
    return {"bounds": {"xmin": x_min, "ymin": y_min, "zmin": z_min}}, {
        "PTS": _get_point_maps(seg_map, CLASSES["KITTI_360"], SCALES["KITTI_360"]),
        "INS": ins_map,
        "SEG": seg_map,
        "TD_HF": tpd_hf,
        "BU_HF": btu_hf,
    }


def _get_kitti_360_merged_projections(metadata, projections):
    # Determine the image bound according to the metadata
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf
    for k, v in metadata.items():
        x_min = min(x_min, v["bounds"]["xmin"])
        y_min = min(y_min, v["bounds"]["ymin"])
        z_min = min(z_min, v["bounds"]["zmin"])
        x_max = max(x_max, v["bounds"]["xmin"] + projections[k]["TD_HF"].shape[1])
        y_max = max(y_max, v["bounds"]["ymin"] + projections[k]["TD_HF"].shape[0])

    merged_metadata = {
        "bounds": {"xmin": int(x_min), "ymin": int(y_min), "zmin": int(z_min)}
    }
    merged_projections = {}
    h, w = y_max - y_min + 1, x_max - x_min + 1
    for k, v in projections.items():
        _h, _w = v["TD_HF"].shape
        _y, _x = (
            metadata[k]["bounds"]["ymin"] - y_min,
            metadata[k]["bounds"]["xmin"] - x_min,
        )
        _z = metadata[k]["bounds"]["zmin"] - z_min
        merged_projections[k] = {
            "PTS": np.zeros((h, w), dtype=bool),
            "INS": np.zeros((h, w), dtype=np.int16),
            "SEG": np.zeros((h, w), dtype=np.int16),
            "TD_HF": np.zeros((h, w), dtype=np.int16),
            "BU_HF": np.zeros((h, w), dtype=np.int16),
        }
        merged_projections[k]["PTS"][_y : _y + _h, _x : _x + _w] = v["PTS"]
        merged_projections[k]["INS"][_y : _y + _h, _x : _x + _w] = v["INS"]
        merged_projections[k]["SEG"][_y : _y + _h, _x : _x + _w] = v["SEG"]
        merged_projections[k]["TD_HF"][_y : _y + _h, _x : _x + _w] = v["TD_HF"] + _z
        merged_projections[k]["BU_HF"][_y : _y + _h, _x : _x + _w] = v["BU_HF"]

    return merged_metadata, merged_projections


def get_seg_map_from_ins_map(dataset, ins_map):
    if dataset == "CITY_SAMPLE":
        return _get_city_sample_seg_map(ins_map)
    elif dataset == "GOOGLE_EARTH":
        return _get_google_earth_seg_map(ins_map)
    elif dataset == "KITTI_360":
        return _get_kitti_360_seg_map(ins_map)
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
    # NOTE: BLDG_ROOF is mapped to BLDG_FACADE because the BLDG_ROOF is not in the semantic map.
    ins_map = ins_map.copy()
    # ins_map[
    #     np.where((ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]) & (ins_map % 2))
    # ] = CLASSES["GOOGLE_EARTH"]["BLDG_ROOF"]
    ins_map[ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]] = CLASSES["GOOGLE_EARTH"][
        "BLDG_FACADE"
    ]
    return ins_map


def _get_kitti_360_seg_map(ins_map):
    # NOTE: BLDG_ROOF is mapped to BLDG_FACADE because the BLDG_ROOF is not in the semantic map.
    ins_map = ins_map.copy()
    ins_map[ins_map >= CONSTANTS["KITTI_360"]["CAR_INS_MIN_ID"]] = CLASSES["KITTI_360"][
        "CAR"
    ]
    # ins_map[
    #     np.where((ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]) & (ins_map % 2))
    # ] = CLASSES["KITTI_360"]["BLDG_ROOF"]
    ins_map[ins_map >= CONSTANTS["BLDG_INS_MIN_ID"]] = CLASSES["KITTI_360"][
        "BLDG_FACADE"
    ]
    return ins_map


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
    CATEGORIES = ["CAR", "FWY", "VEGT", "REST"]
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


def get_seg_ins_relations(dataset):
    return {
        # BLDG
        "BLDG_INS_MIN_ID": CONSTANTS["BLDG_INS_MIN_ID"],
        "ROOF_INS_OFFSET": CONSTANTS["ROOF_INS_OFFSET"],
        "BLDG_FACADE_SEMANTIC_ID": CLASSES[dataset]["BLDG_FACADE"],
        "BLDG_ROOF_SEMANTIC_ID": CLASSES[dataset]["BLDG_ROOF"]
        if "BLDG_ROOF" in CLASSES[dataset]
        else CLASSES[dataset]["BLDG_FACADE"],
        # CAR (not used in Google-Earth)
        "CAR_INS_MIN_ID": CONSTANTS[dataset]["CAR_INS_MIN_ID"]
        if "CAR_INS_MIN_ID" in CONSTANTS[dataset]
        else 32767,
        "CAR_SEMANTIC_ID": CLASSES[dataset]["CAR"]
        if "CAR" in CLASSES[dataset]
        else 32767,
    }


def get_camera_parameters(dataset, city_dir, metadata):
    if dataset == "CITY_SAMPLE":
        return get_city_sample_camera_parameters(city_dir)
    elif dataset == "GOOGLE_EARTH":
        return get_google_earth_camera_parameters(city_dir, metadata)
    elif dataset == "KITTI_360":
        return get_kitti_360_camera_parameters(city_dir, metadata)
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


def get_google_earth_camera_parameters(city_dir, metadata):
    ge_project_name = os.path.basename(city_dir)
    with open(os.path.join(city_dir, "%s.json" % ge_project_name)) as f:
        cam_settings = json.load(f)

    # Intrinsic matrix
    vfov = cam_settings["cameraFrames"][0]["fovVertical"]
    # The MAGIC NUMBER to make it aligned with Google Earth Renderings
    cam_focal = cam_settings["height"] / 2 / np.tan(np.deg2rad(vfov)) * 2.06
    cam_rig = {
        "intrinsics": [
            cam_focal / (960 / CONSTANTS["GOOGLE_EARTH"]["IMAGE_WIDTH"]),
            0,
            cam_settings["width"] // 2,
            0,
            cam_focal / (540 / CONSTANTS["GOOGLE_EARTH"]["IMAGE_HEIGHT"]),
            cam_settings["height"] // 2,
            0,
            0,
            1,
        ],
        "sensor_size": [cam_settings["width"], cam_settings["height"]],
    }
    # Extrinsic matrix
    camera_poses = []
    for f_idx, cf in enumerate(tqdm(cam_settings["cameraFrames"])):
        # Camera position
        tx, ty = _lnglat2xy(
            cf["coordinate"]["longitude"],
            cf["coordinate"]["latitude"],
            metadata["resolution"],
            CONSTANTS["GOOGLE_EARTH"]["ZOOM_LEVEL"],
            dtype=float,
        )
        tx -= metadata["bounds"]["xmin"]
        ty -= metadata["bounds"]["ymin"]
        # Camera rotation
        quat = _get_quat_from_look_at(
            {"x": tx, "y": ty, "z": cf["coordinate"]["altitude"]}, metadata["target"]
        )
        camera_poses.append(
            {
                "id": f_idx,
                "tx": tx - metadata["target"]["x"],
                "ty": ty - metadata["target"]["y"],
                "tz": cf["coordinate"]["altitude"],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
            }
        )
    return cam_rig, camera_poses


def _get_quat_from_look_at(cam_pos, cam_look_at):
    fwd_vec = np.array(
        [
            cam_look_at["x"] - cam_pos["x"],
            cam_look_at["y"] - cam_pos["y"],
            cam_look_at["z"] - cam_pos["z"],
        ]
    )
    fwd_vec /= np.linalg.norm(fwd_vec)
    up_vec = np.array([0, 0, 1])
    right_vec = np.cross(up_vec, fwd_vec)
    right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(fwd_vec, right_vec)
    R = np.stack([fwd_vec, right_vec, up_vec], axis=1)
    return scipy.spatial.transform.Rotation.from_matrix(R).as_quat()


def get_kitti_360_camera_parameters(city_dir, metadata):
    # Ref: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py#L96
    # Intrinsic matrix
    cam_rig = {"intrinsics": [], "sensor_size": []}
    cam_calib_dir = os.path.join(city_dir, os.pardir, os.pardir, "calibration")
    with open(os.path.join(cam_calib_dir, "perspective.txt")) as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.split(" ")
        if line[0] == "P_rect_00:":
            cam_rig["intrinsics"] = (
                np.array([float(x) for x in line[1:]]).reshape((3, 4))[:3, :3].flatten()
            )
        elif line[0] == "S_rect_00:":
            cam_rig["sensor_size"] = [int(float(line[1])), int(float(line[2]))]
    # Fix cx because the coordinate system is mirrored
    cam_rig["intrinsics"][2] = cam_rig["sensor_size"][0] - cam_rig["intrinsics"][2]

    # Extrinsic matrix
    camera_poses = []
    camera_frames = np.loadtxt(os.path.join(city_dir, "cam0_to_world.txt"))
    for cf in tqdm(camera_frames):
        f_idx = int(cf[0])
        Rt = cf[1:].reshape((4, 4))
        # R -> [Right | Down | Forward]
        R = Rt[:3, :3]
        # R -> [Forward | Right | Up]
        R = R[:, [2, 0, 1]]
        # R[:, -1] *= -1
        quat = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
        t = Rt[:3, 3]
        camera_poses.append(
            {
                "id": f_idx,
                "tx": t[0] / CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
                - metadata["bounds"]["xmin"],
                "ty": t[1] / CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
                - metadata["bounds"]["ymin"],
                "tz": t[2] / CONSTANTS["KITTI_360"]["VOXEL_SIZE"]
                - metadata["bounds"]["zmin"],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
            }
        )
    return cam_rig, camera_poses


def save_camera_poses(output_file_path, cam_poses):
    with open(output_file_path, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "id",
                "tx",
                "ty",
                "tz",
                "qx",
                "qy",
                "qz",
                "qw",
            ],
        )
        writer.writeheader()
        writer.writerows(cam_poses)


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
    x4 = (bl2 - be3) / (ke3 - kl2) if not math.isinf(ke3) else x3
    y4 = kl2 * x4 + bl2
    if not math.isinf(ke3):
        assert abs(y4 - ke3 * x4 - be3) < 1e-5
    x5 = (bl4 - be3) / (ke3 - kl4) if not math.isinf(ke3) else x3
    y5 = kl4 * x5 + bl4
    if not math.isinf(ke3):
        assert abs(y5 - ke3 * x5 - be3) < 1e-5
    assert abs((x4 + x5) / 2 - x3) < 1e-5, abs((x4 + x5) / 2 - x3)
    assert abs((y4 + y5) / 2 - y3) < 1e-5, abs((y4 + y5) / 2 - y3)
    # (x6, y6) is the center of the rectangle
    x6, y6 = (x1 + x3) / 2, (y1 + y3) / 2
    # (x7, y7) and (x8, y8) are the two endpoints of E1, respectively.
    x7, y7 = 2 * x6 - x4, 2 * y6 - y4
    x8, y8 = 2 * x6 - x5, 2 * y6 - y5
    assert abs((x7 + x8) / 2 - x1) < 1e-5
    assert abs((y7 + y8) / 2 - y1) < 1e-5
    return np.array([(x1, y1), (x4, y4), (x5, y5), (x7, y7), (x8, y8)], dtype=np.int16)


def get_local_projections(projections, local_cords, map_size):
    MAPS = [
        {"name": "SEG", "dtype": np.uint8, "interpolation": cv2.INTER_NEAREST},
        {"name": "TD_HF", "dtype": np.float32, "interpolation": cv2.INTER_AREA},
    ]

    local_projections = {m["name"]: projections[m["name"]].copy() for m in MAPS}
    if local_cords is not None:
        points = np.array([local_cords[1], local_cords[2], local_cords[0]])
        cx, cy = np.mean(points, axis=0).astype(np.int32)
        # Crop the image
        x_min, x_max = cx - map_size // 2, cx + map_size // 2
        y_min, y_max = cy - map_size // 2, cy + map_size // 2
        for m in MAPS:
            m_name = m["name"]
            m_type = m["dtype"]
            # Fix: _src.total() > 0 in function 'warpPerspective'
            if x_min < 0:
                local_projections[m_name] = np.pad(
                    local_projections[m_name],
                    ((0, 0), (-x_min, 0)),
                    mode="constant",
                    constant_values=0,
                )
                x_max -= x_min
                x_min = 0
            if y_min < 0:
                local_projections[m_name] = np.pad(
                    local_projections[m_name],
                    ((-y_min, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                y_max -= y_min
                y_min = 0
            local_projections[m_name] = local_projections[m_name][
                y_min:y_max, x_min:x_max
            ].astype(m_type)
        # The top-left point of the local projection
        local_projections["tlp"] = np.array([x_min, y_min])

    for m in MAPS:
        m_name = m["name"]
        m_type = m["dtype"]
        m_intp = m["interpolation"]
        local_projections[m_name] = cv2.resize(
            local_projections[m_name].astype(m_type),
            (map_size, map_size),
            interpolation=m_intp,
        )
    return local_projections


def get_points_from_projections(
    projections, classes, scales, seg_ins_relation, water_z, local_cords=None
):
    # XYZ, Scale, Instance ID
    points = np.empty((0, 5), dtype=np.int16)
    for c, p in projections.items():
        # Ignore bottom points for objects in the rest maps due to invisibility.
        _points = _get_points_from_projection(
            p, classes, scales, seg_ins_relation, local_cords, c != "REST"
        )
        if _points is not None:
            points = np.concatenate((points, _points), axis=0)
            logging.debug(
                "Category: %s: #Points: %d, Min/Max Value: (%d, %d)"
                % (c, len(_points), np.min(_points), np.max(_points))
            )
        # Move the water plane to -3.5m, which is aligned with CitySample.
        if c == "REST" and "WATER" in classes:
            points[:, 2][points[:, 4] == classes["WATER"]] = water_z

    logging.debug("#Points: %d" % (len(points)))
    return points


def _get_points_from_projection(
    projection,
    classes,
    scales,
    seg_ins_relation,
    local_cords=None,
    include_btm_pts=True,
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
        # Fix: negative index is not supported. Also aligned with the operations in get_local_projections()
        if min_x < 0:
            max_x -= min_x
            min_x = 0
        if min_y < 0:
            max_y -= min_y
            min_y = 0

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
        seg_ins_relation,
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


def get_sky_points(far_plane, cam_z, cam_fov_y, patch_size, scale, class_id):
    points = []
    # Determine the border of sky
    sky_height = patch_size * math.tan(cam_fov_y)
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
    # NOTE: The point IDs start from 1 to avoid the conflict with the NULL class.
    assert points.shape[0] < 2147483648
    pt_ids = torch.arange(
        start=1, end=points.shape[0] + 1, dtype=torch.int32, device=points.device
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
            cam_rig["intrinsics"][5],
            cam_rig["intrinsics"][2],
        ],
        [cam_rig["sensor_size"][1], cam_rig["sensor_size"][0]],
        N_MAX_SAMPLES,
    )
    # NOTE: The point ID for NULL class is -1, the rest point IDs are from 0 to N - 1.
    # The ray_voxel_intersection_perspective seems not accepting the negative values.
    return voxel_id.squeeze() - 1


def get_visible_points(
    points, scales, cam_rig, cam_pos, cam_quat, null_class_id=0, reduce_mem=False
):
    # NOTE: Each point is assigned with a unique ID. The values in the rendered map
    # denotes the visibility of the points. The values are the same as the point IDs.
    instances = torch.from_numpy(points[:, 4]).cuda()
    points = torch.from_numpy(points[:, [0, 1, 2]]).cuda()
    scales = (
        torch.from_numpy(scales).cuda()
        if isinstance(scales, np.ndarray)
        else scales.cuda()
    )
    # Scale the volume by 0.33 to reduce the memory usage
    if reduce_mem:
        SCALE_FACTOR = 1 / 3.0
        cam_pos *= SCALE_FACTOR
        scales = (scales * SCALE_FACTOR).clamp(min=1).short()
        points = torch.floor(points * SCALE_FACTOR).short()

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
    null_mask = vp_map == -1
    ins_map[null_mask] = null_class_id
    # Image.fromarray(
    #     utils.helpers.get_ins_seg_map.r_palatte[ins_map.cpu().numpy()],
    # ).save("output/test.jpg")

    # Manually release the memory to avoid OOM
    del volume
    torch.cuda.empty_cache()

    return vp_map.cpu().numpy(), ins_map.cpu().numpy()


def main(dataset, data_dir, osm_dir, is_debug):
    assert dataset in ["GOOGLE_EARTH", "KITTI_360", "CITY_SAMPLE"], (
        "Unknown dataset: %s" % dataset
    )

    if dataset == "KITTI_360":
        logging.info("Reorganzing the KITTI 360 dataset ...")
        data_dir = reorganize_kitti_360(data_dir)

    cities = sorted(os.listdir(data_dir))
    for c_idx, city in enumerate(tqdm(cities)):
        logging.info("Generating point projections...")
        city_dir = os.path.join(data_dir, city)
        # The metadata is used for the GOOGLE_EARTH and KITTI_360 dataset
        metadata, projections = get_projections(dataset, city_dir, osm_dir)

        logging.info("Saving projections...")
        proj_dir = os.path.join(city_dir, "Projection")
        dump_projections(projections, proj_dir, is_debug)
        if metadata is not None:
            with open(os.path.join(proj_dir, "metadata.json"), "w") as fp:
                json.dump(metadata, fp)

        # # Debug: Load projection caches without computing
        # logging.info("loading projections...")
        # proj_dir = os.path.join(city_dir, "Projection")
        # projections = load_projections(proj_dir)
        # if os.path.exists(os.path.join(proj_dir, "metadata.json")):
        #     with open(os.path.join(proj_dir, "metadata.json"), "r") as fp:
        #         metadata = json.load(fp)

        logging.info("Calculate the XY center for instances...")
        if os.path.exists(os.path.join(city_dir, "CENTERS.pkl")):
            with open(os.path.join(city_dir, "CENTERS.pkl"), "rb") as fp:
                centers = pickle.load(fp)
        else:
            centers = get_centers_from_projections(projections)
            with open(os.path.join(city_dir, "CENTERS.pkl"), "wb") as fp:
                pickle.dump(centers, fp)

        # Construct the relationship between instance ID and semantic ID
        seg_ins_relation = get_seg_ins_relations(dataset)
        # # Debug: Generate all initial points (casues OOM in rasterization)
        # logging.info("Generate the initial points for the whole city...")
        # # points[:, M] -> 0:3: XYZ, 3: Scale, 4: Instance ID
        # points = get_points_from_projections(
        #     projections,
        #     CLASSES[dataset],
        #     SCALES[dataset],
        #     seg_ins_relation,
        #     CONSTANTS[dataset]["WATER_Z"] if "WATER_Z" in CONSTANTS[dataset] else 0,
        # )

        # # Debug: Point Cloud Visualization
        # logging.info("Saving the generated point cloud...")
        # xyz = points[:, :3]
        # rgbs = utils.helpers.get_ins_colors(points[:, 4])
        # utils.helpers.dump_ptcloud_ply("/tmp/points.ply", xyz, rgbs)

        # Load camera parameters
        cam_rig, cam_poses = get_camera_parameters(dataset, city_dir, metadata)
        # Save the camera parameters for the GoogleEarth dataset
        if dataset in ["GOOGLE_EARTH", "KITTI_360"]:
            save_camera_poses(os.path.join(city_dir, "CameraPoses.csv"), cam_poses)

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
        city_instance_map_dir = os.path.join(city_dir, "InstanceImage")
        os.makedirs(city_points_dir, exist_ok=True)
        os.makedirs(city_instance_map_dir, exist_ok=True)
        for r in tqdm(cam_poses, desc="Rendering Gaussian Points"):
            cam_quat = np.array([r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float32)
            cam_pos = (
                np.array([r["tx"], r["ty"], r["tz"]], dtype=np.float32)
                / CONSTANTS[dataset]["SCALE"]
            )
            cam_pos[0] += CONSTANTS[dataset]["MAP_SIZE"] // 2
            cam_pos[1] += CONSTANTS[dataset]["MAP_SIZE"] // 2
            cam_look_at = utils.helpers.get_camera_look_at(cam_pos, cam_quat)
            logging.debug("Current Camera: %s, Look at: %s" % (cam_pos, cam_look_at))
            # Make sure that the projection patches are with the same sizes
            # For the CitySample dataset, the an affine transformation is applied to
            # the projection patches.
            view_frustum_cords = (
                get_view_frustum_cords(
                    cam_pos,
                    cam_look_at,
                    CONSTANTS[dataset]["PATCH_SIZE"],
                    fov_x / 2,
                )
                if dataset in ["CITY_SAMPLE", "KITTI_360"]
                else None
            )
            local_projections = get_local_projections(
                projections["REST"],
                view_frustum_cords,
                CONSTANTS[dataset]["PROJECTION_SIZE"],
            )
            points = get_points_from_projections(
                projections,
                CLASSES[dataset],
                SCALES[dataset],
                seg_ins_relation,
                CONSTANTS[dataset]["WATER_Z"] if "WATER_Z" in CONSTANTS[dataset] else 0,
                view_frustum_cords,
            )
            # Generate sky points for the CitySample dataset
            if dataset in ["CITY_SAMPLE", "KITTI_360"]:
                sky_points = get_sky_points(
                    view_frustum_cords[1:3],
                    cam_pos[2],
                    fov_y / 2,
                    CONSTANTS[dataset]["PATCH_SIZE"],
                    SCALES[dataset]["SKY"],
                    CLASSES[dataset]["SKY"],
                )
                points = np.concatenate((points, sky_points), axis=0)
            # Generate the instance segmentation map as a side product
            scales = utils.helpers.get_point_scales(points[:, [3]], points[:, [4]])
            vp_map, ins_map = get_visible_points(
                points,
                scales,
                cam_rig,
                cam_pos.copy(),
                cam_quat,
                CLASSES[dataset]["NULL"],
                dataset == "CITY_SAMPLE",
            )

            if dataset == "KITTI_360":
                vp_map = np.fliplr(vp_map)
                ins_map = np.fliplr(ins_map)

            vp_idx = np.sort(np.unique(vp_map))
            vp_idx = vp_idx[vp_idx >= 0]
            # Remove the points that are not visible in the current view.
            # NOTE: The negative value (-1) denotes the NULL class
            points = points[vp_idx]
            # Debug: Visualize the visible points
            # utils.helpers.dump_ptcloud_ply(
            #     "/tmp/points.ply",
            #     points[:, :3],
            #     utils.helpers.get_ins_colors(points[:, 4]),
            # )
            logging.debug("%d points in frame %d." % (len(points), int(r["id"])))
            # Re-generate the visible points map in the newly indexed points
            vp_map = np.searchsorted(vp_idx, vp_map)
            assert np.max(vp_map) == len(points) - 1
            assert len(np.unique(vp_map)) == len(points)
            # # Debug: Render Instance Map with Gaussian Splatting
            # # Revert the cx value for the KITTI 360 dataset. Otherwise, the image cannot be aligned perfectly.
            # _cam_rig = copy.deepcopy(cam_rig)
            # if dataset == "KITTI_360":
            #     _cam_rig["intrinsics"][2] = (
            #         cam_rig["sensor_size"][0] - cam_rig["intrinsics"][2]
            #     )
            # scales = scales[vp_idx]
            # import extensions.diff_gaussian_rasterization as dgr
            # gr = dgr.GaussianRasterizerWrapper(
            #     np.array(_cam_rig["intrinsics"], dtype=np.float32).reshape((3, 3)),
            #     _cam_rig["sensor_size"],
            #     flip_lr=True,
            #     flip_ud=dataset == "KITTI_360",
            #     device=torch.device("cuda"),
            # )
            # with torch.no_grad():
            #     gs_points = utils.helpers.get_gaussian_points(
            #         torch.from_numpy(points[:, :3]).unsqueeze(0),
            #         scales.unsqueeze(0) * 0.5,
            #         {
            #             "rgb": torch.from_numpy(
            #                 utils.helpers.get_ins_colors(points[:, 4])
            #             ).unsqueeze(0)
            #             / 255.0
            #         },
            #     ).cuda()
            #     img = gr(gs_points.squeeze(0), cam_pos, cam_quat)
            #     cv2.imwrite(
            #         "output/test.jpg",
            #         img.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255,
            #     )

            Image.fromarray(ins_map.astype(np.uint16)).save(
                os.path.join(
                    city_instance_map_dir,
                    "%s.png"
                    % (CONSTANTS[dataset]["OUT_FILE_NAME_PATTERN"] % int(r["id"])),
                )
            )
            seg_file_name = (
                CONSTANTS[dataset]["SEG_MAP_PATTERN"] % (city, int(r["id"]))
                if dataset in ["CITY_SAMPLE", "GOOGLE_EARTH"]
                else CONSTANTS[dataset]["SEG_MAP_PATTERN"] % int(r["id"])
            )
            seg_map = np.array(
                Image.open(os.path.join(city_dir, seg_file_name)).convert("P")
            )
            with open(
                os.path.join(
                    city_points_dir,
                    "%s.pkl"
                    % (CONSTANTS[dataset]["OUT_FILE_NAME_PATTERN"] % int(r["id"])),
                ),
                "wb",
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
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CITY_SAMPLE")
    parser.add_argument(
        "--data_dir", default=os.path.join(PROJECT_HOME, "data", "city-sample")
    )
    parser.add_argument("--osm_dir", default=os.path.join(PROJECT_HOME, "data", "osm"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.data_dir, args.osm_dir, args.debug)
