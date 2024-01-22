# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2024-01-18 11:45:08
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-22 10:39:51
# @Email:  root@haozhexie.com

import argparse
import copy
import csv
import cv2
import json
import logging
import numpy as np
import os
import sys
import torch

from PIL import Image
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import models.gancraft
import scripts.dataset_generator
import utils.helpers

CONSTANTS = {
    "IMAGE_WIDTH": 1920,
    "IMAGE_HEIGHT": 1080,
    "IMAGE_PADDING": 8,
    "LAYOUT_MAX_HEIGHT": 384,
    "LAYOUT_N_CLASSES": 9,
    "LAYOUT_VOL_SIZE": 1536,
    "BUILDING_VOL_SIZE": 672,
    "N_MAX_BUILDINGS": 5000,
    "BLD_INS_STEP": 4,
    "BLD_INS_MIN_ID": 100,
    "CAR_INS_MIN_ID": 5000,
    "BLD_FACADE_CLS_ID": 7,
    "BLD_ROOF_CLS_ID": 8,
}


def get_models(gancraft_bg_ckpt, gancraft_fg_ckpt):
    # Load checkpoints
    logging.info("Loading checkpoints ...")
    gancraft_bg_ckpt = torch.load(gancraft_bg_ckpt)
    gancraft_fg_ckpt = torch.load(gancraft_fg_ckpt)

    # Initialize models
    gancraft_bg = models.gancraft.GanCraftGenerator(gancraft_bg_ckpt["cfg"])
    gancraft_fg = models.gancraft.GanCraftGenerator(gancraft_fg_ckpt["cfg"])
    if torch.cuda.is_available():
        gancraft_bg = torch.nn.DataParallel(gancraft_bg).cuda()
        gancraft_fg = torch.nn.DataParallel(gancraft_fg).cuda()
    else:
        gancraft_bg.output_device = torch.device("cpu")
        gancraft_fg.output_device = torch.device("cpu")

    # Recover from checkpoints
    logging.info("Recovering from checkpoints ...")
    gancraft_bg.load_state_dict(gancraft_bg_ckpt["gancraft_g"], strict=False)
    gancraft_fg.load_state_dict(gancraft_fg_ckpt["gancraft_g"], strict=False)

    return gancraft_bg, gancraft_fg


def _get_z(device, z_dim=256):
    if z_dim is None:
        return None

    return torch.randn(1, z_dim, dtype=torch.float32, device=device)


def get_latent_codes(n_buildings, bg_style_dim, output_device):
    bg_z = _get_z(output_device, bg_style_dim)
    building_zs = {
        i: _get_z(output_device)
        for i in range(
            CONSTANTS["BLD_INS_MIN_ID"],
            CONSTANTS["BLD_INS_MIN_ID"] + n_buildings,
            CONSTANTS["BLD_INS_STEP"],
        )
    }
    return bg_z, building_zs


def get_hf_seg_tensor(part_hf, part_seg, output_device):
    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(output_device)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(output_device)
    part_hf = part_hf / CONSTANTS["LAYOUT_MAX_HEIGHT"]
    part_seg = utils.helpers.masks_to_onehots(
        part_seg[:, 0, :, :], CONSTANTS["LAYOUT_N_CLASSES"]
    )
    return torch.cat([part_hf, part_seg], dim=1)


def get_pad_img_bbox(sx, ex, sy, ey):
    psx = sx - CONSTANTS["IMAGE_PADDING"] if sx != 0 else 0
    psy = sy - CONSTANTS["IMAGE_PADDING"] if sy != 0 else 0
    pex = (
        ex + CONSTANTS["IMAGE_PADDING"]
        if ex != CONSTANTS["IMAGE_WIDTH"]
        else CONSTANTS["IMAGE_WIDTH"]
    )
    pey = (
        ey + CONSTANTS["IMAGE_PADDING"]
        if ey != CONSTANTS["IMAGE_HEIGHT"]
        else CONSTANTS["IMAGE_HEIGHT"]
    )
    return psx, pex, psy, pey


def get_img_without_pad(img, sx, ex, sy, ey, psx, pex, psy, pey):
    if CONSTANTS["IMAGE_PADDING"] == 0:
        return img

    return img[
        :,
        :,
        sy - psy : ey - pey if ey != pey else ey,
        sx - psx : ex - pex if ex != pex else ex,
    ]


def render_bg(
    patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_origin, z
):
    _voxel_id = copy.deepcopy(voxel_id)
    _voxel_id[_voxel_id >= CONSTANTS["BLD_INS_MIN_ID"]] = CONSTANTS["BLD_FACADE_CLS_ID"]
    assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    bg_img = torch.zeros(
        1,
        3,
        CONSTANTS["IMAGE_HEIGHT"],
        CONSTANTS["IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_bg.output_device,
    )
    # Render background patches by patch to avoid OOM
    for i in range(CONSTANTS["IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["IMAGE_WIDTH"] // patch_size[1]):
            print("bg", i, j)
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(sx, ex, sy, ey)
            output_bg = gancraft_bg(
                hf_seg=hf_seg,
                voxel_id=_voxel_id[:, psy:pey, psx:pex],
                depth2=depth2[:, psy:pey, psx:pex],
                raydirs=raydirs[:, psy:pey, psx:pex],
                cam_origin=cam_origin,
                footprint_bboxes=None,
                z=z,
                deterministic=True,
            )
            bg_img[:, :, sy:ey, sx:ex] = get_img_without_pad(
                output_bg, sx, ex, sy, ey, psx, pex, psy, pey
            )

    return bg_img


def _get_img_patch(img, cx, cy):
    size = CONSTANTS["BUILDING_VOL_SIZE"]
    half_size = size // 2
    pad_img = torch.zeros(
        size=(img.size(0), img.size(1), size, size), device=img.device
    )
    # Determine the crop position
    tl_x, br_x = cx - half_size, cx + half_size
    tl_y, br_y = cy - half_size, cy + half_size
    # Handle Corner case (out of bounds)
    pad_x = 0 if tl_x >= 0 else abs(tl_x)
    tl_x = tl_x if tl_x >= 0 else 0
    br_x = min(br_x, CONSTANTS["LAYOUT_VOL_SIZE"])
    patch_w = br_x - tl_x
    pad_y = 0 if tl_y >= 0 else abs(tl_y)
    tl_y = tl_y if tl_y >= 0 else 0
    br_y = min(br_y, CONSTANTS["LAYOUT_VOL_SIZE"])
    patch_h = br_y - tl_y
    # Copy-paste
    pad_img[:, :, pad_y : pad_y + patch_h, pad_x : pad_x + patch_w] = img[
        :, :, tl_y:br_y, tl_x:br_x
    ]
    return pad_img


def render_fg(
    patch_size,
    gancraft_fg,
    building_id,
    hf_seg,
    voxel_id,
    depth2,
    raydirs,
    cam_origin,
    footprint_bbox,
    building_z,
):
    _voxel_id = copy.deepcopy(voxel_id)
    _curr_bld = torch.tensor([building_id, building_id + 1], device=voxel_id.device)
    _voxel_id[~torch.isin(_voxel_id, _curr_bld)] = 0
    _voxel_id[voxel_id == building_id] = CONSTANTS["BLD_FACADE_CLS_ID"]
    _voxel_id[voxel_id == building_id + 1] = CONSTANTS["BLD_ROOF_CLS_ID"]

    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    _hf_seg = copy.deepcopy(hf_seg)
    _hf_seg[hf_seg != building_id] = 0
    _hf_seg[hf_seg == building_id] = CONSTANTS["BLD_FACADE_CLS_ID"]
    _raydirs = copy.deepcopy(raydirs)
    _raydirs[_voxel_id[..., 0, 0] == 0] = 0

    # Crop the "hf_seg" image using the center of the target building as the reference
    cx = CONSTANTS["LAYOUT_VOL_SIZE"] // 2 + int(footprint_bbox[1])
    cy = CONSTANTS["LAYOUT_VOL_SIZE"] // 2 + int(footprint_bbox[0])
    _hf_seg = _get_img_patch(hf_seg, cx, cy)

    fg_img = torch.zeros(
        1,
        3,
        CONSTANTS["IMAGE_HEIGHT"],
        CONSTANTS["IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    fg_mask = torch.zeros(
        1,
        1,
        CONSTANTS["IMAGE_HEIGHT"],
        CONSTANTS["IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )

    # Render foreground patches by patch to avoid OOM
    for i in range(CONSTANTS["IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["IMAGE_WIDTH"] // patch_size[1]):
            print("fg", i, j)
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(sx, ex, sy, ey)

            if torch.count_nonzero(_raydirs[:, sy:ey, sx:ex]) > 0:
                output_fg = gancraft_fg(
                    _hf_seg,
                    _voxel_id[:, psy:pey, psx:pex],
                    depth2[:, psy:pey, psx:pex],
                    _raydirs[:, psy:pey, psx:pex],
                    cam_origin,
                    footprint_bboxes=torch.from_numpy(
                        np.array(footprint_bbox)
                    ).unsqueeze(dim=0),
                    z=building_z,
                    deterministic=True,
                )
                facade_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id
                ).unsqueeze(dim=1)
                roof_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id - 1
                ).unsqueeze(dim=1)
                facade_img = facade_mask * get_img_without_pad(
                    output_fg, sx, ex, sy, ey, psx, pex, psy, pey
                )
                roof_img = roof_mask * get_img_without_pad(
                    output_fg,
                    sx,
                    ex,
                    sy,
                    ey,
                    psx,
                    pex,
                    psy,
                    pey,
                )
                fg_mask[:, :, sy:ey, sx:ex] = torch.logical_or(facade_mask, roof_mask)
                fg_img[:, :, sy:ey, sx:ex] = (
                    facade_img * facade_mask + roof_img * roof_mask
                )

    return fg_img, fg_mask


def render(
    patch_size,
    seg_volume,
    hf_seg,
    cam_rig,
    cam_pose,
    gancraft_bg,
    gancraft_fg,
    footprint_bboxes,
    bg_z,
    building_zs,
):
    raycasting = scripts.dataset_generator.get_ray_voxel_intersection(
        cam_rig, cam_pose["cam_position"], cam_pose["cam_look_at"], seg_volume
    )
    voxel_id = raycasting["voxel_id"].unsqueeze(dim=0)
    depth2 = raycasting["depth2"].permute(1, 2, 0, 3, 4).unsqueeze(dim=0)
    raydirs = raycasting["raydirs"].unsqueeze(dim=0)
    cam_origin = raycasting["cam_origin"].unsqueeze(dim=0)

    buildings = torch.unique(voxel_id[voxel_id > CONSTANTS["BLD_INS_MIN_ID"]])
    # Remove odd numbers from the list because they are reserved by roofs.
    buildings = buildings[buildings % CONSTANTS["BLD_INS_STEP"] == 0]
    with torch.no_grad():
        bg_img = render_bg(
            patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_origin, bg_z
        )
        for b in buildings:
            assert (
                b % CONSTANTS["BLD_INS_STEP"] == 0
            ), "Building Instance ID MUST be an even number."
            fg_img, fg_mask = render_fg(
                patch_size,
                gancraft_fg,
                b.item(),
                hf_seg,
                voxel_id,
                depth2,
                raydirs,
                cam_origin,
                footprint_bboxes[b.item()],
                building_zs[b.item()],
            )
            bg_img = bg_img * (1 - fg_mask) + fg_img * fg_mask

    return bg_img


def get_video(frames, output_file):
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"avc1"),
        4,
        (CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"]),
    )
    for f in frames:
        video.write(f)

    video.release()


def main(patch_size, output_file, gancraft_bg_ckpt, gancraft_fg_ckpt):
    gancraft_bg, gancraft_fg = get_models(gancraft_bg_ckpt, gancraft_fg_ckpt)
    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z, building_zs = get_latent_codes(
        CONSTANTS["N_MAX_BUILDINGS"],
        gancraft_bg.module.cfg.NETWORK.GANCRAFT.STYLE_DIM,
        gancraft_bg.output_device,
    )

    # Generate the concatenated height field and seg. layout tensor
    city_name = "City01"
    height_field = Image.open(
        os.path.join(PROJECT_HOME, "data", city_name, "HeightField.png")
    )
    seg_layout = Image.open(
        os.path.join(PROJECT_HOME, "data", city_name, "SegLayout.png")
    )
    height_field = np.array(height_field)
    seg_layout = np.array(seg_layout)

    hf_seg = get_hf_seg_tensor(height_field, seg_layout, gancraft_bg.output_device)
    # print(hf_seg.size())    # torch.Size([1, 10, 1536, 1536])
    footprint_bboxes = scripts.dataset_generator.get_footprint_bboxes(seg_layout)

    # Build seg_volume
    logging.info("Generating seg volume ...")
    height_field = torch.from_numpy(height_field).to(gancraft_bg.output_device)
    seg_layout = torch.from_numpy(seg_layout).to(gancraft_bg.output_device)
    seg_volume = scripts.dataset_generator.get_volume_with_roof_1f(
        height_field, seg_layout
    )

    # Generate camera trajectories
    logging.info("Generating camera poses ...")
    with open(os.path.join(PROJECT_HOME, "data", city_name, "CameraRig.json")) as fp:
        cam_rig = json.load(fp)
        cam_rig = cam_rig["cameras"]["CameraComponent"]

    cam_poses = []
    with open(os.path.join(PROJECT_HOME, "data", city_name, "CameraPoses.csv")) as fp:
        reader = csv.DictReader(fp)
        cam_poses = [
            scripts.dataset_generator.get_camera_poses(r, seg_volume.size())
            for r in reader
        ]

    logging.info("Rendering videos ...")
    frames = []
    for cam_pose in tqdm(cam_poses):
        img = render(
            patch_size,
            seg_volume,
            hf_seg,
            cam_rig,
            cam_pose,
            gancraft_bg,
            gancraft_fg,
            footprint_bboxes,
            bg_z,
            building_zs,
        )
        img = (utils.helpers.tensor_to_image(img, "RGB") * 255).astype(np.uint8)
        frames.append(img[..., ::-1])
        cv2.imwrite("output/test.jpg", img[..., ::-1])

    get_video(frames, output_file)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gancraft_bg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "gancraft-bg.pth"),
    )
    parser.add_argument(
        "--gancraft_fg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "gancraft-fg.pth"),
    )
    parser.add_argument(
        "--patch_height",
        default=CONSTANTS["IMAGE_HEIGHT"] // 10,
        type=int,
    )
    parser.add_argument(
        "--patch_width",
        default=CONSTANTS["IMAGE_WIDTH"] // 10,
        type=int,
    )
    parser.add_argument(
        "--output_file",
        default=os.path.join(PROJECT_HOME, "output", "rendering.mp4"),
        type=str,
    )
    args = parser.parse_args()

    main(
        (args.patch_height, args.patch_width),
        args.output_file,
        args.gancraft_bg_ckpt,
        args.gancraft_fg_ckpt,
    )
