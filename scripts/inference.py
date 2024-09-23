# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2024-01-18 11:45:08
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-09-23 20:35:31
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

from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import extensions.diff_gaussian_rasterization as dgr
import models.generator
import scripts.dataset_generator
import utils.helpers

CONSTANTS = {
    "GOOGLE_EARTH": {
        "N_CLASSES": 8,
        "N_TRAJECTORY_POINTS": 24,
        "POINT_SCALE_FACTOR": 0.65,
        "SPECIAL_Z_SCALE_CLASSES": {"ROAD": 1, "WATER": 5, "ZONE": 6},
        "INST_RANGE": {"REST": [0, 10], "BLDG": [100, 16384]},
        "PROJ_SIZE": 2048,
        "SENSOR_SIZE": (960, 540),
        "K": [1528.1469407006614, 0, 480, 0, 1528.1469407006614, 270, 0, 0, 1],
    },
    "KITTI_360": {
        "N_CLASSES": 8,
        "POINT_SCALE_FACTOR": 0.65,
        "SPECIAL_Z_SCALE_CLASSES": {"ROAD": 1, "ZONE": 6},
        "INST_RANGE": {"REST": [0, 10], "BLDG": [100, 10000], "CAR": [10000, 16384]},
        "PATCH_SIZE": 1280,
        "PROJ_SIZE": 2048,
        "SENSOR_SIZE": (1408, 376),
        "K": [552.554261, 0, 682.049453, 0, 552.554261, 238.769549, 0, 0, 1],
    },
}


def _get_model(dataset, ckpt_file_path):
    if not os.path.exists(ckpt_file_path):
        return None

    ckpt = torch.load(ckpt_file_path, weights_only=False)
    model = models.generator.Generator(
        ckpt["cfg"].NETWORK.GAUSSIAN,
        CONSTANTS[dataset]["N_CLASSES"],
        CONSTANTS[dataset]["PROJ_SIZE"],
    )
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.output_device = torch.device("cpu")

    model.load_state_dict(ckpt["gaussian_g"], strict=False)
    if "module.z" in ckpt["gaussian_g"]:
        model.module.z = ckpt["gaussian_g"]["module.z"]

    return model


def get_models(dataset, bldg_ckpt, car_ckpt, rest_ckpt):
    rest_model = _get_model(dataset, rest_ckpt)

    bldg_model = None
    if bldg_ckpt is not None:
        bldg_model = _get_model(dataset, bldg_ckpt)

    car_model = None
    if car_ckpt is not None:
        car_model = _get_model(dataset, car_ckpt)

    return bldg_model, car_model, rest_model


def get_city_projections(dataset_dir):
    cities = sorted(
        [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
    )
    city = np.random.choice(cities)
    city_dir = os.path.join(dataset_dir, city)

    proj_dir = os.path.join(city_dir, "Projection")
    projections = scripts.dataset_generator.load_projections(proj_dir)
    metadata = None
    if os.path.exists(os.path.join(proj_dir, "metadata.json")):
        with open(os.path.join(proj_dir, "metadata.json"), "r") as fp:
            metadata = json.load(fp)
            metadata["city_dir"] = city_dir

    with open(os.path.join(city_dir, "CENTERS.pkl"), "rb") as fp:
        centers = pickle.load(fp)

    return metadata, projections, centers


def get_style_lut(centers, models, inst_range, z_dim=256):
    lut = {ins: torch.rand(1, z_dim) for ins in centers.keys()}
    for k, v in models.items():
        if v is None:
            continue

        if v.module.cfg.Z_DIM is None:
            for i in range(*inst_range[k]):
                if i in lut:
                    del lut[i]
            continue

        keys = [k for k in centers.keys()]
        if hasattr(v.module, "z"):
            zs = v.module.z
            lut.update(
                {ins: zs[ins].unsqueeze(0) for ins in keys}
                # {ins: zs[np.random.choice(keys)].unsqueeze(0) for ins in keys}
            )

    return lut


def get_camera_poses(dataset, metadata):
    if dataset == "GOOGLE_EARTH":
        return _get_google_earth_camera_poses()
    elif dataset == "KITTI_360":
        return _get_kitti_360_camera_poses(metadata["city_dir"])
    else:
        raise NotImplementedError


def _get_google_earth_camera_poses():
    radius = np.random.randint(384, 768)
    altitude = np.random.randint(384, 768)
    logging.info("Radius = %d, Altitude = %s" % (radius, altitude))
    cx = CONSTANTS["GOOGLE_EARTH"]["PROJ_SIZE"] // 2
    cy = CONSTANTS["GOOGLE_EARTH"]["PROJ_SIZE"] // 2

    camera_poses = []
    for i in range(CONSTANTS["GOOGLE_EARTH"]["N_TRAJECTORY_POINTS"]):
        theta = 2 * math.pi / CONSTANTS["GOOGLE_EARTH"]["N_TRAJECTORY_POINTS"] * i
        cam_x = cx + radius * math.cos(theta)
        cam_y = cy + radius * math.sin(theta)

        quat = scripts.dataset_generator._get_quat_from_look_at(
            {"x": cam_x, "y": cam_y, "z": altitude},
            {"x": cx, "y": cy, "z": 1},
        )
        camera_poses.append(
            {
                "id": i,
                "tx": cam_x,
                "ty": cam_y,
                "tz": altitude,
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
            }
        )
    return camera_poses


def _get_kitti_360_camera_poses(city_dir):
    with open(os.path.join(city_dir, "CameraPoses.csv")) as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def render(dataset, projections, centers, style_lut, cam_pose, gr, models):
    cam_quat = np.array(
        [cam_pose["qx"], cam_pose["qy"], cam_pose["qz"], cam_pose["qw"]],
        dtype=np.float32,
    )
    cam_pos = np.array(
        [cam_pose["tx"], cam_pose["ty"], cam_pose["tz"]], dtype=np.float32
    )
    local_projections, pts = _get_bev_points(dataset, projections, cam_pos, cam_quat)

    pts, batch_idx = _get_normalized_pt_cords(pts, centers)
    (
        batch_idx,
        pts,
        proj_hf,
        proj_seg,
        proj_tlp,
    ) = _get_tensors(dataset, batch_idx, pts, local_projections)

    instances = pts[:, :, [4]]
    classes = _instances_to_classes(dataset, instances)
    scales = utils.helpers.get_point_scales(
        pts[:, :, [3]] * CONSTANTS[dataset]["POINT_SCALE_FACTOR"],
        classes,
        CONSTANTS[dataset]["SPECIAL_Z_SCALE_CLASSES"].values(),
    )

    bldg_idx, car_idx, rest_idx = _get_pt_indexes_by_models(
        classes, models["BLDG"], models["CAR"]
    )
    abs_xyz, scales, pt_attrs = _get_pt_attrs_by_models(
        dataset,
        batch_idx,
        pts,
        scales,
        classes,
        instances,
        style_lut,
        {"TD_HF": proj_hf, "SEG": proj_seg, "TLP": proj_tlp},
        {"BLDG": bldg_idx, "CAR": car_idx, "REST": rest_idx},
        models,
    )

    gs_pts = utils.helpers.get_gaussian_points(abs_xyz, scales, pt_attrs)
    with torch.no_grad():
        fake_img = utils.helpers.get_gaussian_rasterization(
            gs_pts,
            gr,
            cam_pos[None, ...],
            cam_quat[None, ...],
        )
    return fake_img


def _get_bev_points(dataset, projections, cam_pos, cam_quat):
    fov_x = utils.helpers.intrinsic_to_fov(
        CONSTANTS[dataset]["K"][0], CONSTANTS[dataset]["SENSOR_SIZE"][0]
    )
    fov_y = utils.helpers.intrinsic_to_fov(
        CONSTANTS[dataset]["K"][4], CONSTANTS[dataset]["SENSOR_SIZE"][1]
    )
    cam_look_at = utils.helpers.get_camera_look_at(cam_pos, cam_quat)
    view_frustum_cords = (
        scripts.dataset_generator.get_view_frustum_cords(
            cam_pos,
            cam_look_at,
            CONSTANTS[dataset]["PATCH_SIZE"],
            fov_x / 2,
        )
        if dataset in ["CITY_SAMPLE", "KITTI_360"]
        else None
    )

    local_projections = scripts.dataset_generator.get_local_projections(
        projections["REST"],
        view_frustum_cords,
        CONSTANTS[dataset]["PROJ_SIZE"],
    )
    points = scripts.dataset_generator.get_points_from_projections(
        projections,
        scripts.dataset_generator.CLASSES[dataset],
        scripts.dataset_generator.SCALES[dataset],
        scripts.dataset_generator.get_seg_ins_relations(dataset),
        CONSTANTS[dataset]["WATER_Z"] if "WATER_Z" in CONSTANTS[dataset] else 0,
        view_frustum_cords,
    )
    # Generate sky points for the CitySample dataset
    if dataset in ["CITY_SAMPLE", "KITTI_360"]:
        sky_points = scripts.dataset_generator.get_sky_points(
            view_frustum_cords[1:3],
            cam_pos[2],
            fov_y / 2,
            CONSTANTS[dataset]["PATCH_SIZE"],
            scripts.dataset_generator.SCALES[dataset]["SKY"],
            scripts.dataset_generator.CLASSES[dataset]["SKY"],
        )
        points = np.concatenate((points, sky_points), axis=0)
    # Generate the instance segmentation map as a side product
    scales = utils.helpers.get_point_scales(points[:, [3]], points[:, [4]])
    vp_map, ins_map = scripts.dataset_generator.get_visible_points(
        points,
        scales,
        {
            "intrinsics": CONSTANTS[dataset]["K"],
            "sensor_size": CONSTANTS[dataset]["SENSOR_SIZE"],
        },
        cam_pos.copy(),
        cam_quat,
        scripts.dataset_generator.CLASSES[dataset]["NULL"],
        dataset == "CITY_SAMPLE",
    )
    if dataset == "KITTI_360":
        vp_map = np.fliplr(vp_map)
        ins_map = np.fliplr(ins_map)

    vp_idx = np.sort(np.unique(vp_map))
    vp_idx = vp_idx[vp_idx >= 0]
    points = points[vp_idx]
    return local_projections, points


def _get_normalized_pt_cords(pts, centers):
    instances = np.unique(pts[:, -1])
    rel_cords = pts[:, :3].copy().astype(np.float32)
    batch_idx = np.zeros((pts.shape[0], 1), dtype=np.int32)
    for idx, ins in enumerate(instances):
        is_pts = pts[:, -1] == ins
        cx, cy, w, h, d = centers[ins]

        rel_cords[is_pts, 0] = (pts[is_pts, 0] - cx) / w * 2 if w > 0 else 0
        rel_cords[is_pts, 1] = (pts[is_pts, 1] - cy) / h * 2 if h > 0 else 0
        rel_cords[is_pts, 2] = (
            np.clip(pts[is_pts, 2] / d * 2 - 1, -1, 1) if d > 0 else 0
        )
        batch_idx[is_pts, 0] = idx

    return np.concatenate((pts, rel_cords), axis=1), batch_idx


def _get_tensors(dataset, batch_idx, pts, local_projections):
    batch_idx = utils.helpers.var_or_cuda(torch.from_numpy(batch_idx[None, ...]))
    pts = utils.helpers.var_or_cuda(torch.from_numpy(pts[None, ...]))
    proj_hf = utils.helpers.var_or_cuda(
        torch.from_numpy(local_projections["TD_HF"][None, None, ...])
    )
    proj_seg = utils.helpers.var_or_cuda(
        torch.from_numpy(
            _get_onehot_seg(local_projections["SEG"], CONSTANTS[dataset]["N_CLASSES"])[
                None, ...
            ]
        ).float()
    )
    proj_tlp = (
        utils.helpers.var_or_cuda(torch.from_numpy(local_projections["tlp"][None, ...]))
        if "tlp" in local_projections
        else None
    )
    return (
        batch_idx,
        pts,
        proj_hf,
        proj_seg,
        proj_tlp,
    )


def _get_onehot_seg(mask, n_classes):
    h, w = mask.shape
    one_hot_masks = np.zeros((n_classes, h, w), dtype=np.uint8)
    for i in range(n_classes):
        one_hot_masks[i] = mask == i

    return one_hot_masks


def _get_z(instances, style_lut):
    b, n, c = instances.size()
    assert b == 1 and c == 1, "Unexpected tensor shape (%d, %d, %d)" % (b, n, c)

    unique_instances = [i.item() for i in torch.unique(instances).short()]
    unique_z = {
        ui: style_lut[ui].to(instances.device)
        for ui in unique_instances
        if ui in style_lut
    }
    # The style code is disabled for these instances
    if not unique_z:
        return None

    z = {}
    for ui in unique_instances:
        idx = instances[..., 0] == ui
        z[ui] = {
            "z": unique_z[ui],
            "idx": idx,
        }
    return z


def _get_pt_indexes_by_models(classes, bldg_model, car_model):
    classes = classes.squeeze()
    car_idx = torch.zeros_like(classes)
    bldg_idx = torch.zeros_like(classes)
    if bldg_model is not None:
        bldg_idx = torch.isin(
            classes,
            torch.tensor(
                [
                    scripts.dataset_generator.CLASSES["GOOGLE_EARTH"]["BLDG_FACADE"],
                    scripts.dataset_generator.CLASSES["GOOGLE_EARTH"]["BLDG_ROOF"],
                ],
                device=classes.device,
            ),
        )
    if car_model is not None:
        car_idx = torch.isin(
            classes,
            torch.tensor(
                [
                    scripts.dataset_generator.CLASSES["GOOGLE_EARTH"]["BLDG_FACADE"],
                    scripts.dataset_generator.CLASSES["GOOGLE_EARTH"]["BLDG_ROOF"],
                ],
                device=classes.device,
            ),
        )
    rest_idx = ~torch.logical_or(bldg_idx, car_idx)
    return bldg_idx, car_idx, rest_idx


def _get_pt_attrs_by_models(
    dataset,
    batch_idx,
    pts,
    scales,
    classes,
    instances,
    style_lut,
    projections,
    indexes,
    models,
):
    reordered_abs_xyz = []
    reordered_scales = []
    reordered_pt_attrs = {}
    for k in models.keys():
        idx = indexes[k]
        model = models[k]
        if torch.sum(idx) == 0 or model is None:
            continue

        # Make batch_idx contiguous and starts from 0
        _batch_idx = batch_idx[:, idx].clone()
        _batch_idxes = torch.unique(_batch_idx)
        for i, bi in enumerate(_batch_idxes):
            _batch_idx[_batch_idx == bi] = i

        pt_attrs = _get_gaussian_attributes(
            dataset,
            _batch_idx[..., 0],
            pts[:, idx],
            classes[:, idx],
            _get_z(instances[:, idx], style_lut),
            projections["TD_HF"],
            projections["SEG"],
            projections["TLP"],
            model,
        )
        reordered_abs_xyz.append(pts[:, idx, :3])
        reordered_scales.append(scales[:, idx])
        for k, v in pt_attrs.items():
            if k not in reordered_pt_attrs:
                reordered_pt_attrs[k] = []
            reordered_pt_attrs[k].append(v)

    for k, v in reordered_pt_attrs.items():
        reordered_pt_attrs[k] = torch.cat(v, dim=1)

    return (
        torch.cat(reordered_abs_xyz, dim=1),
        torch.cat(reordered_scales, dim=1),
        reordered_pt_attrs,
    )


def _get_gaussian_attributes(
    dataset,
    batch_idx,
    pts,
    classes,
    zs,
    proj_hf,
    proj_seg,
    proj_tlp,
    model,
):
    abs_xyz = pts[:, :, :3]
    rel_xyz = pts[:, :, 5:8]
    onehots = utils.helpers.get_one_hot(classes, CONSTANTS[dataset]["N_CLASSES"])
    proj_uv = utils.helpers.get_projection_uv(
        abs_xyz,
        proj_tlp,
        CONSTANTS[dataset]["PROJ_SIZE"],
    )
    with torch.no_grad():
        pt_attrs = model(proj_uv, rel_xyz, batch_idx, onehots, zs, proj_hf, proj_seg)

    return pt_attrs


def _instances_to_classes(dataset, instances):
    if dataset == "GOOGLE_EARTH":
        return _google_earth_instances_to_classes(instances)
    elif dataset == "KITTI_360":
        return _kitti_360_instances_to_classes(instances)
    else:
        raise NotImplementedError


def _google_earth_instances_to_classes(instances):
    bldg_facade_idx = (
        instances >= scripts.dataset_generator.CONSTANTS["BLDG_INS_MIN_ID"]
    ) & (instances % 2 == 0)
    bldg_roof_idx = (
        instances >= scripts.dataset_generator.CONSTANTS["BLDG_INS_MIN_ID"]
    ) & (instances % 2 == 1)

    classes = instances.clone()
    classes[bldg_facade_idx] = scripts.dataset_generator.CLASSES["GOOGLE_EARTH"][
        "BLDG_FACADE"
    ]
    classes[bldg_roof_idx] = scripts.dataset_generator.CLASSES["GOOGLE_EARTH"][
        "BLDG_ROOF"
    ]
    return classes


def _kitti_360_instances_to_classes(instances):
    bldg_facade_idx = (
        (instances >= scripts.dataset_generator.CONSTANTS["BLDG_INS_MIN_ID"])
        & (
            instances
            < scripts.dataset_generator.CONSTANTS["KITTI_360"]["CAR_INS_MIN_ID"]
        )
        & (instances % 2 == 0)
    )
    bldg_roof_idx = (
        (instances >= scripts.dataset_generator.CONSTANTS["BLDG_INS_MIN_ID"])
        & (
            instances
            < scripts.dataset_generator.CONSTANTS["KITTI_360"]["CAR_INS_MIN_ID"]
        )
        & (instances % 2 == 1)
    )
    car_idx = (
        instances >= scripts.dataset_generator.CONSTANTS["KITTI_360"]["CAR_INS_MIN_ID"]
    )

    classes = instances.clone()
    classes[bldg_facade_idx] = scripts.dataset_generator.CLASSES["KITTI_360"][
        "BLDG_FACADE"
    ]
    classes[bldg_roof_idx] = scripts.dataset_generator.CLASSES["KITTI_360"]["BLDG_ROOF"]
    classes[car_idx] = scripts.dataset_generator.CLASSES["KITTI_360"]["CAR"]
    return classes


def get_video(frames, img_size, output_file):
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"avc1"),
        4,
        (img_size[0], img_size[1]),  # (width, height)
    )
    for f in frames:
        video.write(f)

    video.release()


def main(dataset, dataset_dir, output_file, bldg_ckpt, car_ckpt, rest_ckpt):
    logging.info("Loading checkpoints ...")
    bldg_model, car_model, rest_model = get_models(
        dataset, bldg_ckpt, car_ckpt, rest_ckpt
    )

    logging.info("Generating city layout ...")
    metadata, projections, centers = get_city_projections(dataset_dir)

    logging.info("Generating style look-up table ...")
    style_lut = get_style_lut(
        centers,
        {
            "BLDG": bldg_model,
            "CAR": car_model,
            "REST": rest_model,
        },
        CONSTANTS[dataset]["INST_RANGE"],
    )

    logging.info("Generating camera poses ...")
    cam_poses = get_camera_poses(dataset, metadata)

    logging.info("Rendering videos ...")
    gr = dgr.GaussianRasterizerWrapper(
        np.array(CONSTANTS[dataset]["K"], dtype=np.float32).reshape((3, 3)),
        CONSTANTS[dataset]["SENSOR_SIZE"],
        flip_lr=True,
        flip_ud=dataset == "KITTI_360",
        device=torch.device("cuda"),
    )
    frames = []
    for f_idx, cam_pose in enumerate(tqdm(cam_poses)):
        img = render(
            dataset,
            projections,
            centers,
            style_lut,
            cam_pose,
            gr,
            {"BLDG": bldg_model, "CAR": car_model, "REST": rest_model},
        )
        img = (utils.helpers.tensor_to_image(img, "RGB") * 255).astype(np.uint8)
        frames.append(img[..., ::-1])
        cv2.imwrite("output/render/%04d.jpg" % f_idx, img[..., ::-1])

    get_video(frames, CONSTANTS[dataset]["SENSOR_SIZE"], output_file)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GOOGLE_EARTH")
    parser.add_argument(
        "--data_dir", default=os.path.join(PROJECT_HOME, "data", "google-earth")
    )
    parser.add_argument(
        "--bldg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "bldg.pth"),
    )
    parser.add_argument(
        "--car_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "car.pth"),
    )
    parser.add_argument(
        "--rest_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "rest.pth"),
    )
    parser.add_argument(
        "--output_file",
        default=os.path.join(PROJECT_HOME, "output", "rendering.mp4"),
        type=str,
    )
    args = parser.parse_args()
    main(
        args.dataset,
        args.data_dir,
        args.output_file,
        args.bldg_ckpt,
        args.car_ckpt,
        args.rest_ckpt,
    )
