# -*- coding: utf-8 -*-
#
# @File:   transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-09-18 22:08:13
# @Email:  root@haozhexie.com

import numpy as np
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


class RandomCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.mode = parameters["mode"] if "mode" in parameters else "random"
        self.n_min_pixels = (
            parameters["n_min_pixels"] if "n_min_pixels" in parameters else 0
        )
        self.n_min_points = (
            parameters["n_min_points"] if "n_min_points" in parameters else 0
        )
        self.n_max_points = (
            parameters["n_max_points"] if "n_max_points" in parameters else 0
        )
        self.objects = objects

    def _get_offset(self, size, crop_size):
        if size == crop_size:
            return 0
        elif self.mode == "random":
            return np.random.randint(0, size - crop_size - 1)
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
                if self.n_max_points == 0 and self.n_min_points == 0:
                    break

                n_points = len(np.unique(visible_pts))
                if (self.n_min_points == 0 or n_points >= self.n_min_points) and (
                    self.n_max_points == 0 or n_points <= self.n_max_points
                ):
                    break
        # else:
        #     offset_x, offset_y = None, None

        return offset_x, offset_y, mask, visible_pts

    def __call__(self, data):
        width, height = self.width, self.height
        offset_x, offset_y = None, None
        while offset_x is None or offset_y is None:
            offset_x, offset_y, mask, visible_pts = self._get_crop_position(
                data, width, height
            )
            # width /= 2
            # height /= 2

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


class RandomInstance(object):
    def __init__(self, parameters, objects):
        self.range = parameters["range"] if "range" in parameters else None
        self.n_instances = (
            parameters["n_instances"] if "n_instances" in parameters else 1
        )
        self.objects = objects

    def __call__(self, data):
        ins_map = data["ins"] * data["msk"]
        visible_ins = np.unique(ins_map[ins_map > 0])
        if self.range is not None:
            visible_ins = visible_ins[visible_ins >= self.range[0]]
            visible_ins = visible_ins[visible_ins < self.range[1]]

        if len(visible_ins) == 0:
            data["msk"] = np.zeros_like(data["msk"])
            return data

        ins = (
            np.random.choice(visible_ins, self.n_instances, replace=False)
            if self.n_instances > 0
            else visible_ins
        )
        ins_mask = np.isin(data["ins"], ins)

        data["msk"] &= ins_mask
        data["vpm"][~data["msk"]] = -1
        return data


class RemoveUnseenPoints(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        vpm = data["vpm"]
        visible_pts = np.unique(vpm[vpm != -1])
        data["pts"] = data["pts"][visible_pts]
        return data


class NormalizePointCords(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        instances = np.unique(data["pts"][:, -1])
        rel_cords = data["pts"][:, :3].copy().astype(np.float32)
        batch_idx = np.zeros((data["pts"].shape[0], 1), dtype=np.float32)
        for idx, ins in enumerate(instances):
            is_pts = data["pts"][:, -1] == ins
            cx, cy, w, h, d = data["centers"][ins]

            rel_cords[is_pts, 0] = (data["pts"][is_pts, 0] - cx) / w * 2 if w > 0 else 0
            rel_cords[is_pts, 1] = (data["pts"][is_pts, 1] - cy) / h * 2 if h > 0 else 0
            rel_cords[is_pts, 2] = (
                np.clip(data["pts"][is_pts, 2] / d * 2 - 1, -1, 1) if d > 0 else 0
            )
            batch_idx[is_pts, 0] = idx

        data["pts"] = np.concatenate((data["pts"], rel_cords, batch_idx), axis=1)
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
