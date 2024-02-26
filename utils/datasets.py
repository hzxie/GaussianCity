# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-26 10:58:05
# @Email:  root@haozhexie.com

import numpy as np
import os
import random
import torch

import utils.io
import utils.transforms

from tqdm import tqdm


def get_dataset(cfg, dataset_name, split):
    if dataset_name == "CITY_SAMPLE":
        return CitySampleDataset(cfg, split)
    elif dataset_name == "GOOGLE_EARTH":
        raise NotImplementedError()
    else:
        raise Exception("Unknown dataset: %s" % dataset_name)


def collate_fn(batch):
    data = {}
    for sample in batch:
        for k, v in sample.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        if type(v[0]) == torch.Tensor:
            data[k] = torch.stack(v, 0)
        else:
            data[k] = v

    return data


class CitySampleDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(CitySampleDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.memcached = {}
        self.renderings = self._get_renderings(cfg, split)
        self.n_renderings = len(self.renderings)
        self.transforms = self._get_data_transforms(cfg, split)

    def __len__(self):
        return (
            self.n_renderings * self.cfg.DATASETS.CITY_SAMPLE.N_REPEAT
            if self.split == "train"
            else self.n_renderings
        )

    def __getitem__(self, idx):
        idx = idx % self.n_renderings
        rendering = self.renderings[idx]

        K = (
            self.memcached["K"]
            if "K" in self.memcached
            else utils.io.IO.get(rendering["K"])
        )
        Rt = (
            self.memcached["Rt"]
            if "Rt" in self.memcached
            else utils.io.IO.get(rendering["Rt"])
        )
        centers = (
            self.memcached["centers"]
            if "centers" in self.memcached
            else utils.io.IO.get(rendering["centers"])
        )
        rgb = np.array(utils.io.IO.get(rendering["rgb"]), dtype=np.float32)
        rgb = rgb / 255.0 * 2 - 1
        pts = utils.io.IO.get(rendering["pts"])
        data = {
            "K": K["cameras"]["CameraComponent"]["intrinsics"],
            "Rt": Rt[idx],
            "centers": centers,
            "rgb": rgb,
            "ins": pts["ins"],
            "msk": pts["msk"],
            "pts": pts["pts"],
        }
        data = self.transforms(data)
        return data

    def _get_renderings(self, cfg, split):
        cities = [
            "City%02d" % (i + 1) for i in range(cfg.DATASETS.CITY_SAMPLE.N_CITIES)
        ]
        files = [
            {
                "name": "%s/%s/%04d" % (c, s, i),
                # Camera parameters
                "K": os.path.join(cfg.DATASETS.CITY_SAMPLE.DIR, c, "CameraRig.json"),
                "Rt": os.path.join(cfg.DATASETS.CITY_SAMPLE.DIR, c, "CameraPoses.csv"),
                # The XY centers of the instances
                "centers": os.path.join(cfg.DATASETS.CITY_SAMPLE.DIR, c, "CENTERS.pkl"),
                "rgb": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR,
                    c,
                    "ColorImage",
                    s,
                    "%sSequence.%04d.jpeg" % (c, i),
                ),
                # Precomputed Points in the viewpoint (scripts/dataset_generator.py)
                "pts": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR, c, "Points", "%04d.pkl" % i
                ),
            }
            for c in cities
            for i in range(cfg.DATASETS.CITY_SAMPLE.N_VIEWS)
            for s in cfg.DATASETS.CITY_SAMPLE.CITY_STYLES
        ]
        if not cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY:
            return files

        for f in tqdm(files, desc="Loading partial files to RAM"):
            for k, v in f.items():
                if k not in cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY:
                    continue
                elif v in self.memcached:
                    continue
                else:
                    self.memcached[v] = utils.io.IO.get(v)

        return files if split == "train" else files[-32:]

    def _get_data_transforms(self, cfg, split):
        if split == "train":
            return utils.transforms.Compose(
                [
                    {
                        "callback": "Crop",
                        "parameters": {
                            "height": cfg.TRAIN.GAUSSIAN.CROP_SIZE[1],
                            "width": cfg.TRAIN.GAUSSIAN.CROP_SIZE[0],
                            "n_min_pixels": cfg.DATASETS.CITY_SAMPLE.N_MIN_PIXELS_CROP,
                        },
                        "objects": ["rgb", "ins", "msk"],
                    },
                    {
                        "callback": "RemoveUnseenPoints",
                        "parameters": None,
                        "objects": ["pts", "ins"],
                    },
                    {
                        "callback": "NormalizePointCords",
                        "parameters": None,
                        "objects": ["pts", "centers"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "rgb",
                            "msk",
                            "pts",
                        ],
                    },
                ]
            )
        else:
            return utils.transforms.Compose(
                [
                    {
                        "callback": "Crop",
                        "parameters": {
                            "height": cfg.TEST.GAUSSIAN.CROP_SIZE[1],
                            "width": cfg.TEST.GAUSSIAN.CROP_SIZE[0],
                            "mode": "center",
                        },
                        "objects": ["rgb", "ins", "msk"],
                    },
                    {
                        "callback": "RemoveUnseenPoints",
                        "parameters": None,
                        "objects": ["pts", "ins"],
                    },
                    {
                        "callback": "NormalizePointCords",
                        "parameters": None,
                        "objects": ["pts", "centers"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "rgb",
                            "msk",
                            "pts",
                        ],
                    },
                ]
            )
