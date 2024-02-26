# -*- coding: utf-8 -*-
#
# @File:   transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-26 14:21:47
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
                if k == "msk" or k == "ins":
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
            patch = self._get_img_patch(data["msk"], offset_x, offset_y)
            n_pixels = np.count_nonzero(patch)
            if n_pixels >= self.n_min_pixels:
                break

        # Crop all data fields simultaneously
        data["crop"] = {"x": offset_x, "y": offset_y, "w": self.width, "h": self.height}
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._get_img_patch(v, offset_x, offset_y)

        return data


class RemoveUnseenPoints(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        im_instances = np.unique(data["ins"])
        pt_instances = data["pts"][:, -1]
        data["pts"] = data["pts"][np.isin(pt_instances, im_instances)]
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
            rel_cords[is_pts, 2] = data["pts"][is_pts, 2] / d

        data["pts"] = np.concatenate((data["pts"], rel_cords), axis=1)
        return data
