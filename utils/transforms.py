# -*- coding: utf-8 -*-
#
# @File:   transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-28 15:51:14
# @Email:  root@haozhexie.com

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
                if k == "msk" or k == "vpm":
                    # H, W -> C, H, W
                    v = v[None, ...]
                elif k == "rgb":
                    # H, W, C -> C, H, W
                    v = v.transpose((2, 0, 1))

                data[k] = torch.from_numpy(v).float()

        return data


class Crop(object):
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

    def __call__(self, data):
        N_MAX_TRY_TIMES = 100
        img = data[self.objects[0]]
        h, w = img.shape[0], img.shape[1]
        # Check the cropped patch contains enough informative pixels for training
        for _ in range(N_MAX_TRY_TIMES):
            offset_x = self._get_offset(w, self.width)
            offset_y = self._get_offset(h, self.height)
            mask = self._get_img_patch(data["msk"], offset_x, offset_y)
            visible_pts = self._get_img_patch(data["vpm"], offset_x, offset_y)

            n_pixels = np.count_nonzero(mask)
            if n_pixels >= self.n_min_pixels:
                if self.n_max_points <= 0:
                    break
                n_points = np.unique(visible_pts)
                if len(n_points) <= self.n_max_points:
                    break

        # Crop all data fields simultaneously
        data["crop"] = {"x": offset_x, "y": offset_y, "w": self.width, "h": self.height}
        for k, v in data.items():
            if k == "msk":
                # Prevent duplicated computation
                data[k] = mask
            elif k == "vpm":
                # Prevent duplicated computation
                data[k] = visible_pts
            elif k in self.objects:
                data[k] = self._get_img_patch(v, offset_x, offset_y)

        return data


class RemoveUnseenPoints(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        visible_pts = np.unique(data["vpm"])
        data["pts"] = data["pts"][visible_pts]
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
            rel_cords[is_pts, 2] = np.clip(data["pts"][is_pts, 2] / d, 0, 1)

        data["pts"] = np.concatenate((data["pts"], rel_cords), axis=1)
        return data
