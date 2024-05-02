# -*- coding: utf-8 -*-
#
# @File:   helper.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:25:10
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-05-02 21:37:24
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
    N_MAX_INSTANCES = 16384
    return (
        get_ins_seg_map.r_palatte[obj % N_MAX_INSTANCES].astype(np.uint8)
        if random
        else get_ins_seg_map.f_palatte[obj % N_MAX_INSTANCES].astype(np.uint8)
    )


def get_one_hot(classes, n_class):
    b, n, c = classes.size()
    assert c == 1, "Unexpected tensor shape (%d, %d)" % (n, c)

    one_hot = torch.zeros(b, n, n_class).to(classes.device)
    one_hot.scatter_(2, classes.long(), 1)
    return one_hot


def get_z(instances, z_dim):
    b, n, c = instances.size()
    assert b == 1 and c == 1, "Unexpected tensor shape (%d, %d, %d)" % (b, n, c)

    unique_instances = [i.item() for i in torch.unique(instances).short()]
    unique_z = {
        ui: torch.rand(1, z_dim).to(instances.device) for ui in unique_instances
    }

    z = {}
    for ui in unique_instances:
        idx = instances[..., 0] == ui
        z[ui] = {
            "z": unique_z[ui],
            "idx": idx,
        }
    return z


def intrinsic_to_fov(focal_length, img_size):
    return 2 * np.arctan2(img_size, (2 * focal_length))


def get_camera_look_at(cam_position, cam_quaternion, step=1000):
    mat3 = scipy.spatial.transform.Rotation.from_quat(cam_quaternion).as_matrix()
    return cam_position + mat3[:3, 0] * step


def onehot_to_mask(onehot, ignored_classes=[]):
    mask = torch.argmax(onehot, dim=1)
    for ic in ignored_classes:
        mask[mask >= ic] += 1

    return mask


def repeat_pts(pts, repeat=1):
    b, n, _ = pts.size()
    pts = pts.repeat(1, repeat, 1)
    idx = torch.arange(repeat, device=pts.device) / repeat
    idx = idx.unsqueeze(dim=0).unsqueeze(dim=-1).repeat(b, n, 1)
    return torch.cat([pts, idx], dim=-1)


def get_projection_uv(xyz, proj_tlp, proj_aff_mat, proj_size):
    n_pts = xyz.size(1)
    if proj_aff_mat is None or proj_tlp is None:
        proj_uv = xyz[..., :2].clone()
    else:
        proj_xy1 = torch.cat(
            [
                xyz[..., :2] - proj_tlp.unsqueeze(dim=1),
                torch.ones(1, n_pts, 1, device=xyz.device),
            ],
            dim=-1,
        )
        proj_uv = torch.bmm(proj_aff_mat, proj_xy1.permute(0, 2, 1)).permute(0, 2, 1)[
            ..., :2
        ]

    assert proj_uv.size() == (xyz.size(0), n_pts, 2)
    proj_uv[..., 0] /= proj_size[0]
    proj_uv[..., 1] /= proj_size[1]
    # Normalize to [-1, 1]
    return proj_uv * 2 - 1


def get_point_scales(scales, classes, special_z_scale_classes=[]):
    if isinstance(scales, np.ndarray):
        scales = torch.from_numpy(scales)
    if isinstance(classes, np.ndarray):
        classes = torch.from_numpy(classes)

    repeat = [1 for _ in scales.size()]
    repeat[-1] = 3
    # Apply special scale factors for BLDGs
    # scales[
    #     torch.isin(
    #         classes.squeeze(dim=-1),
    #         torch.tensor(list(bldg_classes), device=classes.device),
    #     )
    # ] * bldg_factor
    scales_3d = torch.ones_like(scales).repeat(repeat) * scales
    # Set the z-scale = 1 for roads, zones, and waters
    scales_3d[..., 2][
        torch.isin(
            classes.squeeze(dim=-1),
            torch.tensor(
                list(special_z_scale_classes),
                device=classes.device,
            ),
        )
    ] = 1
    return scales_3d


def get_gaussian_points(xyz, scales, attrs):
    batch_size = xyz.size(0)
    n_pts = xyz.size(1)

    rgb = attrs["rgb"]
    if "xyz" in attrs:
        xyz += attrs["xyz"]
    if "scale" in attrs:
        scales *= attrs["scale"]
    if "opacity" in attrs:
        opacity = attrs["opacity"]
    else:
        opacity = torch.ones((batch_size, n_pts, 1), device=xyz.device)

    rotations = torch.cat(
        [
            torch.ones(batch_size, n_pts, 1, device=xyz.device),
            torch.zeros(batch_size, n_pts, 3, device=xyz.device),
        ],
        dim=-1,
    )
    return torch.cat((xyz, opacity, scales, rotations, rgb), dim=-1)


def get_gaussian_rasterization(
    gs_points, rasterizator, cam_pos, cam_quat, crop_bboxes=None
):
    images = []
    batch_size = gs_points.size(0)
    for i in range(batch_size):
        img = rasterizator(
            gs_points[i],
            cam_pos[i],
            cam_quat[i],
        )
        if crop_bboxes is not None:
            cbx = crop_bboxes[i]
            img = img[
                :,
                cbx["y"] : cbx["y"] + cbx["h"],
                cbx["x"] : cbx["x"] + cbx["w"],
            ]
        images.append(img)

    return torch.stack(images, dim=0)


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


def tensor_to_image(tensor, mode):
    # assert mode in ["SegMap", "RGB"]
    tensor = tensor.cpu().numpy()
    if mode == "SegMap":
        return get_seg_map(tensor.squeeze()).convert("RGB")
    elif mode == "RGB":
        return tensor.squeeze().transpose((1, 2, 0)) / 2 + 0.5
    elif mode == "Mask":
        return tensor.transpose((1, 2, 0)).squeeze()
    else:
        raise Exception("Unknown mode: %s" % mode)
