# -*- coding: utf-8 -*-
#
# @File:   transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-06-15 15:28:43
# @Email:  root@haozhexie.com

import cv2
import numpy as np
import random
import torch

import utils.helpers


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr["callback"])
            parameters = tr["parameters"] if "parameters" in tr else None
            self.transformers.append(
                {
                    "callback": transformer(
                        parameters, tr["objects"] if "objects" in tr else None
                    ),
                }
            )

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr["callback"]
            data = transform(data)

        return data


class ToTensor(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                if len(v.shape) == 2:
                    # H, W -> H, W, C
                    v = v[..., None]
                if len(v.shape) == 3:
                    # H, W, C -> C, H, W
                    v = v.transpose((2, 0, 1))

                data[k] = torch.from_numpy(v).float()

        return data


class RandomFlip(object):
    def __init__(self, parameters, objects):
        self.hflip = parameters["hflip"] if parameters else True
        self.vflip = parameters["vflip"] if parameters else True
        self.objects = objects

    def _random_flip(self, img, hflip, vflip):
        if hflip:
            img = np.flip(img, axis=1)
        if vflip:
            img = np.flip(img, axis=0)

        return img.copy()

    def __call__(self, data):
        hflip = True if random.random() <= 0.5 and self.hflip else False
        vflip = True if random.random() <= 0.5 and self.vflip else False
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._random_flip(v, hflip, vflip)

        return data


class CenterCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.objects = objects

    def _center_crop(self, img):
        h, w = img.shape[0], img.shape[1]
        offset_x = w // 2 - self.width // 2
        offset_y = h // 2 - self.height // 2
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width
        ]
        return new_img

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._center_crop(v)

        return data


class RandomCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.objects = objects

    def _crop(self, img, offset_x, offset_y):
        new_img = None
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width
        ]
        return new_img

    def __call__(self, data):
        img = data[self.objects[0]]
        h, w = img.shape[0], img.shape[1]
        offset_x = random.randint(0, w - self.width)
        offset_y = random.randint(0, h - self.height)
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._crop(v, offset_x, offset_y)

        return data


class RandomCropTarget(RandomCrop):
    def __init__(self, parameters, objects):
        super(RandomCropTarget, self).__init__(parameters, objects)
        self.VOXEL_ID_KEY = "voxel_id"
        self.target_value = parameters["target_value"]
        self.objects = objects

    def _get_target_bbox(self, voxel_id, target_value):
        mask = voxel_id[..., 0, 0] == target_value
        pts = cv2.findNonZero(mask.astype(np.uint8))
        x_min, x_max = np.min(pts[..., 0]), np.max(pts[..., 0])
        y_min, y_max = np.min(pts[..., 1]), np.max(pts[..., 1])
        return (x_min, x_max), (y_min, y_max)

    def __call__(self, data):
        img = data[self.objects[0]]
        h, w = img.shape[0], img.shape[1]
        x, y = self._get_target_bbox(data[self.VOXEL_ID_KEY], self.target_value)
        cx, cy = random.randint(x[0], x[1]), random.randint(y[0], y[1])
        offset_x = min(max(0, cx - self.width // 2), w - self.width)
        offset_y = min(max(0, cy - self.height // 2), h - self.height)

        for k, v in data.items():
            if k in self.objects:
                data[k] = self._crop(v, offset_x, offset_y)
        return data


class CenterCropTarget(RandomCropTarget):
    def __init__(self, parameters, objects):
        super(CenterCropTarget, self).__init__(parameters, objects)

    def __call__(self, data):
        img = data[self.objects[0]]
        h, w = img.shape[0], img.shape[1]
        x, y = self._get_target_bbox(data[self.VOXEL_ID_KEY], self.target_value)
        cx, cy = (x[0] + x[1]) // 2, (y[0] + y[1]) // 2
        offset_x = min(max(0, cx - self.width // 2), w - self.width)
        offset_y = min(max(0, cy - self.height // 2), h - self.height)

        for k, v in data.items():
            if k in self.objects:
                data[k] = self._crop(v, offset_x, offset_y)
        return data


class BuildingMaskRemap(object):
    def __init__(self, parameters, objects):
        self.attr = parameters["attr"] if "attr" in parameters else None
        self.bld_facade_label = parameters["bld_facade_label"]
        self.bld_roof_label = parameters["bld_roof_label"]
        self.min_bld_ins_id = parameters["min_bld_ins_id"]
        self.objects = objects

    def _building_mask_remap(self, seg_map, value_map):
        if value_map is not None:
            for src, dst in value_map.items():
                seg_map[seg_map == src] = dst

        seg_map[seg_map >= self.min_bld_ins_id] = 0
        return seg_map

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                bld_ins_id = data[self.attr] if self.attr in data else None
                value_map = (
                    {
                        bld_ins_id: self.bld_facade_label,
                        bld_ins_id - 1: self.bld_roof_label,
                    }
                    if bld_ins_id is not None
                    else None
                )
                data[k] = self._building_mask_remap(
                    v,
                    value_map,
                )

        return data


class MaskRaydirs(object):
    def __init__(self, parameters, objects=None):
        self.VOXEL_ID_KEY = "voxel_id"
        self.RAYDIR_KEY = "raydirs"
        self.attr = parameters["attr"]
        self.values = parameters["values"]
        self.objects = objects

    def __call__(self, data):
        seg_map = data[self.VOXEL_ID_KEY][..., 0, 0]
        mask = np.isin(seg_map, self.values)
        data[self.RAYDIR_KEY][~mask] = 0
        return data


class ToOneHot(object):
    def __init__(self, parameters, objects):
        self.n_classes = parameters["n_classes"]
        self.ignored_classes = (
            parameters["ignored_classes"] if "ignored_classes" in parameters else []
        )
        self.objects = objects

    def _to_onehot(self, img):
        mask = utils.helpers.mask_to_onehot(img, self.n_classes, self.ignored_classes)
        return mask

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._to_onehot(v)

        return data
