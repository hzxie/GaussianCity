# -*- coding: utf-8 -*-
#
# @File:   transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-09 19:57:09
# @Email:  root@haozhexie.com

import cv2
import numpy as np
import random
import torch


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
                if k in ["msk", "proj/hf"]:
                    # H, W -> C, H, W
                    v = v[None, ...]
                elif k in ["rgb", "seg", "proj/seg"]:
                    # H, W, C -> C, H, W
                    v = v.transpose((2, 0, 1))

                data[k] = torch.from_numpy(v).float()

        return data


class InstanceToClass(object):
    def __init__(self, parameters, objects):
        self.map = parameters["map"]
        self.objects = objects

    def _instance_to_class(self, mask):
        for m in self.map:
            src, dst = m["src"], m["dst"]
            mask[np.where((mask >= src[0]) & (mask < src[1]))] = dst

        return mask

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._instance_to_class(v)

        return data


class CropAndRotate(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.readonly = parameters["readonly"] if "readonly" in parameters else []
        self.dtype = parameters["dtype"] if "dtype" in parameters else {}
        self.interpolation = (
            parameters["interpolation"] if "interpolation" in parameters else {}
        )
        self.objects = objects

    def _get_crop(self, points):
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        y_min = points[:, 1].min()
        y_max = points[:, 1].max()
        return x_min, x_max, y_min, y_max

    def _get_rotation(self, points):
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

    def __call__(self, data):
        # Refer to scripts/data_generator.py for the data structure
        points = np.array(
            [data["vfc"][1], data["vfc"][2], data["vfc"][3], data["vfc"][4]]
        )
        # Crop the image
        x_min, x_max, y_min, y_max = self._get_crop(points)
        for k, v in data.items():
            if k in self.objects and k not in self.readonly:
                data[k] = v[y_min:y_max, x_min:x_max].astype(self.dtype[k])

        points -= [x_min, y_min]
        # Rotate the image
        M, width, height = self._get_rotation(points)
        for k, v in data.items():
            if k in self.objects and k not in self.readonly:
                data[k] = cv2.resize(
                    cv2.warpPerspective(v, M, (width, height)),
                    (self.width, self.height),
                    interpolation=(
                        self.interpolation[k]
                        if k in self.interpolation
                        else cv2.INTER_LINEAR
                    ),
                )

        data["proj/tlp"] = np.array([x_min, y_min])
        data["proj/affmat"] = M
        return data


class RandomCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.mode = parameters["mode"] if "mode" in parameters else "random"
        self.n_min_pixels = (
            parameters["n_min_pixels"] if "n_min_pixels" in parameters else 0
        )
        self.n_max_points = (
            parameters["n_max_points"] if "n_max_points" in parameters else 0
        )
        self.objects = objects

    def _get_offset(self, size, crop_size):
        if self.mode == "random":
            return random.randint(0, size - crop_size - 1)
        elif self.mode == "center":
            return size // 2 - crop_size // 2
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def _get_img_patch(self, img, offset_x, offset_y):
        return img[offset_y : offset_y + self.height, offset_x : offset_x + self.width]

    def _get_crop_position(self, data, width, height):
        N_MAX_TRY_TIMES = 100
        img = data[self.objects[0]]
        h, w = img.shape[0], img.shape[1]
        # Check the cropped patch contains enough informative pixels for training
        for _ in range(N_MAX_TRY_TIMES):
            offset_x = self._get_offset(w, width)
            offset_y = self._get_offset(h, height)
            mask = self._get_img_patch(data["msk"], offset_x, offset_y)
            visible_pts = self._get_img_patch(data["vpm"], offset_x, offset_y)

            n_pixels = np.count_nonzero(mask)
            if n_pixels >= self.n_min_pixels:
                if self.n_max_points == 0:
                    break
                n_points = len(np.unique(visible_pts))
                if n_points <= self.n_max_points:
                    break
        else:
            offset_x, offset_y = None, None

        return offset_x, offset_y, mask, visible_pts

    def __call__(self, data):
        width, height = self.width, self.height
        offset_x, offset_y = None, None
        while offset_x is None or offset_y is None:
            offset_x, offset_y, mask, visible_pts = self._get_crop_position(
                data, width, height
            )
            width /= 2
            height /= 2

        # Crop all data fields simultaneously
        data["crp"] = {
            "x": offset_x,
            "y": offset_y,
            "w": self.width,
            "h": self.height,
        }
        for k, v in data.items():
            if k == "msk":
                # Prevent duplicated computation
                data[k] = mask
            elif k == "vpm":
                # Prevent duplicated computation
                data[k] = visible_pts
            if k in self.objects:
                data[k] = self._get_img_patch(v, offset_x, offset_y)

        return data


class RemoveUnseenPoints(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        visible_pts = np.unique(data["vpm"])
        data["pts"] = data["pts"][visible_pts]
        data["pts"] = data["pts"]
        return data


class NormalizePointCords(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        instances = np.unique(data["pts"][:, -1])
        rel_cords = data["pts"][:, :3].copy().astype(np.float32)
        for i in instances:
            is_pts = data["pts"][:, -1] == i
            cx, cy, w, h, d = data["centers"][i]

            rel_cords[is_pts, 0] = (data["pts"][is_pts, 0] - cx) / w * 2
            rel_cords[is_pts, 1] = (data["pts"][is_pts, 1] - cy) / h * 2
            rel_cords[is_pts, 2] = np.clip(data["pts"][is_pts, 2] / d * 2 - 1, -1, 1)

        data["pts"] = np.concatenate((data["pts"], rel_cords), axis=1)
        return data


class ToOneHot(object):
    def __init__(self, parameters, objects):
        self.n_classes = parameters["n_classes"]
        self.ignored_classes = (
            parameters["ignored_classes"] if "ignored_classes" in parameters else []
        )
        self.objects = objects

    def _to_onehot(self, mask):
        h, w = mask.shape
        n_class_actual = self.n_classes - len(self.ignored_classes)
        one_hot_masks = np.zeros((h, w, n_class_actual), dtype=np.uint8)

        n_class_cnt = 0
        for i in range(self.n_classes):
            if i not in self.ignored_classes:
                one_hot_masks[..., n_class_cnt] = mask == i
                n_class_cnt += 1

        return one_hot_masks

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._to_onehot(v)

        return data
