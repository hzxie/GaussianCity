# -*- coding: utf-8 -*-
#
# @File:   helper.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:25:10
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-26 20:55:40
# @Email:  root@haozhexie.com

import numpy as np
import plyfile
import scipy.spatial.transform
import torch

from PIL import Image

count_parameters = lambda n: sum(p.numel() for p in n.parameters())


def var_or_cuda(x, device=None):
    x = x.contiguous()
    if torch.cuda.is_available() and device != torch.device("cpu"):
        if device is None:
            x = x.cuda(non_blocking=True)
        else:
            x = x.cuda(device=device, non_blocking=True)
    return x


def requires_grad(model, require=True):
    for p in model.parameters():
        p.requires_grad = require


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def get_seg_map_palette():
    palatte = np.array([[i, i, i] for i in range(256)])
    # fmt: off
    palatte[:9] = np.array(
        [
            [0, 0, 0],       # empty        -> black (ONLY used in voxel)
            [96, 0, 0],      # road         -> red
            [96, 96, 0],     # freeway      -> yellow
            [0, 96, 0],      # car          -> green
            [0, 96, 96],     # water        -> cyan
            [0, 0, 96],      # sky          -> blue
            [96, 96, 96],    # ground       -> gray
            [96, 0, 96],     # building     -> magenta
            [255, 0, 255],   # bldg. roof   -> lime yellow
        ]
    )
    # fmt: on
    return palatte


@static_vars(palatte=get_seg_map_palette())
def get_seg_map(seg_map):
    if np.max(seg_map) >= 9:
        return get_ins_seg_map(seg_map)

    seg_map = Image.fromarray(seg_map.astype(np.uint8))
    seg_map.putpalette(get_seg_map.palatte.reshape(-1).tolist())
    return seg_map


def get_ins_seg_map_palette(legacy_palette, random=True):
    MAX_N_INSTANCES = 16384
    if random:
        # Make sure that the roof colors are similar to the corresponding facade colors.
        # The odd and even indexes are reserved for roof and facade, respectively.
        palatte0 = np.random.randint(256, size=(MAX_N_INSTANCES, 3), dtype=np.uint8)
        palatte1 = palatte0 * 0.6
        palatte = np.concatenate((palatte0, palatte1), axis=1)
        palatte = palatte0
        palatte = palatte.reshape(-1, 3)
        palatte[:9] = legacy_palette[:9]
    else:
        palatte = np.array(
            [
                [i % 4 * 64, i * 4 % 256, (i * 4 // 256) % 256]
                for i in range(MAX_N_INSTANCES)
            ],
            dtype=np.uint8,
        )
    return palatte


def get_ins_id(img):
    # In get_ins_seg_map_palette, the instance IDs are encoded as RGB values.
    # The function converts the RGB values back to the instance IDs.
    instances = img[..., 1] + img[..., 2] * 256
    instances = np.round(instances / 4).astype(np.uint16)
    # Check CRC
    error_idx = np.round(img[..., 0] / 64).astype(np.uint8) != instances % 4
    instances[error_idx] = 0
    return instances


@static_vars(
    r_palatte=get_ins_seg_map_palette(get_seg_map_palette(), random=True),
    f_palatte=get_ins_seg_map_palette(get_seg_map_palette(), random=False),
)
def get_ins_seg_map(seg_map):
    return Image.fromarray(get_ins_colors(seg_map))


def get_ins_colors(obj, random=True):
    # NOTE: The obj can be a seg_map or ptcloud.
    # If random is True, the instance colors are randomly generated.
    # Otherwise, it will be generated based on the object index.
    return (
        get_ins_seg_map.r_palatte[obj].astype(np.uint8)
        if random
        else get_ins_seg_map.f_palatte[obj].astype(np.uint8)
    )


def masks_to_onehots(masks, n_class, ignored_classes=[]):
    b, h, w = masks.shape
    n_class_actual = n_class - len(ignored_classes)
    one_hot_masks = torch.zeros(
        (b, n_class_actual, h, w), dtype=torch.float32, device=masks.device
    )

    n_class_cnt = 0
    for i in range(n_class):
        if i not in ignored_classes:
            one_hot_masks[:, n_class_cnt] = masks == i
            n_class_cnt += 1

    return one_hot_masks


def mask_to_onehot(mask, n_class, ignored_classes=[]):
    h, w = mask.shape
    n_class_actual = n_class - len(ignored_classes)
    one_hot_masks = np.zeros((h, w, n_class_actual), dtype=np.uint8)

    n_class_cnt = 0
    for i in range(n_class):
        if i not in ignored_classes:
            one_hot_masks[..., n_class_cnt] = mask == i
            n_class_cnt += 1

    return one_hot_masks


def onehot_to_mask(onehot, ignored_classes=[]):
    mask = torch.argmax(onehot, dim=1)
    for ic in ignored_classes:
        mask[mask >= ic] += 1

    return mask


def tensor_to_image(tensor, mode):
    # assert mode in ["HeightField", "FootprintCtr", "SegMap", "RGB"]
    tensor = tensor.cpu().numpy()
    if mode == "HeightField":
        return tensor.transpose((1, 2, 0)).squeeze() / np.max(tensor)
    elif mode == "FootprintCtr":
        return tensor.transpose((1, 2, 0)).squeeze()
    elif mode == "SegMap":
        return get_seg_map(tensor.squeeze()).convert("RGB")
    elif mode == "RGB":
        return tensor.squeeze().transpose((1, 2, 0)) / 2 + 0.5
    else:
        raise Exception("Unknown mode: %s" % mode)


def intrinsic_to_fov(focal_length, img_size):
    return 2 * np.arctan2(img_size, (2 * focal_length))


def get_camera_look_at(cam_position, cam_quaternion, step=1000):
    mat3 = scipy.spatial.transform.Rotation.from_quat(cam_quaternion).as_matrix()
    return cam_position + mat3[:3, 0] * step


def get_point_scales(scales, classes):
    CLASSES = {"ROAD": 1, "WATER": 4, "ZONE": 6}
    if isinstance(scales, np.ndarray):
        scales = torch.from_numpy(scales)
    if isinstance(classes, np.ndarray):
        classes = torch.from_numpy(classes)

    scales = (
        torch.ones(classes.shape[0], 3, dtype=torch.float32, device=classes.device)
        * scales
    )
    # Set the z-scale = 1 for roads, zones, and waters
    scales[:, 2][
        torch.isin(
            classes,
            torch.tensor(
                [
                    CLASSES["ROAD"],
                    CLASSES["WATER"],
                    CLASSES["ZONE"],
                ],
                device=classes.device,
            ),
        )
    ] = 1
    return scales


def dump_ptcloud_ply(ply_fpath, xyz, rgb, attrs={}):
    # Automatically align XY-plane to center of the point cloud
    center_x = (np.min(xyz[:, 0]) + np.max(xyz[:, 0])) / 2
    center_y = (np.min(xyz[:, 1]) + np.max(xyz[:, 1])) / 2
    xyz[:, 0] -= center_x.astype(np.int16)
    xyz[:, 1] -= center_y.astype(np.int16)

    pts = [
        (
            xyz[i, 0],
            xyz[i, 1],
            xyz[i, 2],
            rgb[i, 0],
            rgb[i, 1],
            rgb[i, 2],
            *[attrs[k][i] for k in sorted(attrs.keys())],
        )
        for i in range(xyz.shape[0])
    ]
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        *[(k, "f4") for k in sorted(attrs.keys())],
    ]
    plyfile.PlyData(
        [
            plyfile.PlyElement.describe(
                np.array(
                    pts,
                    dtype=dtype,
                ),
                "vertex",
            )
        ]
    ).write(ply_fpath)
