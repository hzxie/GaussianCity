# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-13 18:32:59
# @Email:  root@haozhexie.com

import copy
import numpy as np
import os
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
        if isinstance(v[0], torch.Tensor):
            data[k] = torch.stack(v, 0)
        elif isinstance(v[0], int):
            data[k] = torch.stack([torch.tensor(v)], 0)
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

    def get_K(self):
        return np.array(self.cfg.DATASETS.CITY_SAMPLE.CAM_K, dtype=np.float32).reshape(
            (3, 3)
        )

    def get_n_classes(self):
        return self.cfg.DATASETS.CITY_SAMPLE.N_CLASSES

    def get_special_z_scale_classes(self):
        return list(self.cfg.DATASETS.CITY_SAMPLE.Z_SCALE_SPECIAL_CLASSES.values())

    def get_proj_size(self):
        return self.cfg.DATASETS.CITY_SAMPLE.PROJ_SIZE

    def instances_to_classes(self, instances):
        # Make it compatible in both numpy and PyTorch
        cfg = self.cfg.DATASETS.CITY_SAMPLE
        bldg_facade_idx = (
            (instances >= cfg.BLDG_RANGE[0])
            & (instances < cfg.BLDG_RANGE[1])
            & (instances % 2 == 0)
        )
        bldg_roof_idx = (
            (instances >= cfg.BLDG_RANGE[0])
            & (instances < cfg.BLDG_RANGE[1])
            & (instances % 2 == 1)
        )
        car_idx = (instances >= cfg.CAR_RANGE[0]) & (instances < cfg.CAR_RANGE[1])

        classes = copy.deepcopy(instances)
        classes[bldg_facade_idx] = cfg.BLDG_FACADE_CLSID
        classes[bldg_roof_idx] = cfg.BLDG_ROOF_CLSID
        classes[car_idx] = cfg.CAR_CLSID
        return classes

    def __len__(self):
        return (
            self.n_renderings * self.cfg.DATASETS.CITY_SAMPLE.N_REPEAT
            if self.split == "train"
            else self.n_renderings
        )

    def __getitem__(self, idx):
        rendering = self.renderings[idx % self.n_renderings]
        view_idx = int(rendering["name"].split("/")[-1])

        Rt = (
            self.memcached["Rt"].copy()
            if "Rt" in self.memcached
            else utils.io.IO.get(rendering["Rt"])
        )
        centers = (
            self.memcached["centers"].copy()
            if "centers" in self.memcached
            else utils.io.IO.get(rendering["centers"])
        )

        rgb = np.array(utils.io.IO.get(rendering["rgb"]), dtype=np.float32)
        rgb = rgb / 255.0 * 2 - 1
        seg = np.array(utils.io.IO.get(rendering["seg"]).convert("P"))
        ins = np.array(utils.io.IO.get(rendering["ins"]))
        pts = utils.io.IO.get(rendering["pts"])
        Rt = Rt[view_idx]
        # Normalize the camera position to fit the scale of the map.
        # Matched with the scripts/dataset_generator.py.
        cam_pos = (
            np.array([Rt["tx"], Rt["ty"], Rt["tz"]], dtype=np.float32)
            / self.cfg.DATASETS.CITY_SAMPLE.SCALE
        )
        cam_pos[:2] += self.cfg.DATASETS.CITY_SAMPLE.MAP_SIZE // 2

        data = {
            "cam_pos": cam_pos,
            "cam_quat": np.array(
                [Rt["qx"], Rt["qy"], Rt["qz"], Rt["qw"]], dtype=np.float32
            ),
            "centers": centers,
            "rgb": rgb,
            "seg": seg,
            "ins": ins,
            "proj/hf": pts["prj"]["TD_HF"],
            "proj/seg": pts["prj"]["SEG"],
            "proj/affmat": pts["prj"]["affmat"],
            "proj/tlp": pts["prj"]["tlp"],
            "vpm": pts["vpm"],
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
                "seg": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR,
                    c,
                    "SemanticImage",
                    "%sSequence.%04d.png" % (c, i),
                ),
                "ins": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR,
                    c,
                    "InstanceImage",
                    "%04d.png" % i,
                ),
                # Projection
                "proj/hf": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR, c, "Projection", "REST-TD_HF.png"
                ),
                "proj/seg": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR, c, "Projection", "REST-SEG.png"
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

        return files if split == "train" else files[-16:]

    def _get_data_transforms(self, cfg, split):
        if split == "train":
            return utils.transforms.Compose(
                [
                    {
                        "callback": "RandomCrop",
                        "parameters": {
                            "height": cfg.TRAIN.GAUSSIAN.IMG_CROP_SIZE[1],
                            "width": cfg.TRAIN.GAUSSIAN.IMG_CROP_SIZE[0],
                            "n_min_pixels": cfg.TRAIN.GAUSSIAN.N_MIN_PIXELS,
                            "n_max_points": cfg.TRAIN.GAUSSIAN.N_MAX_POINTS,
                        },
                        "objects": ["rgb", "seg", "ins", "vpm", "msk"],
                    },
                    {
                        "callback": "RandomInstance",
                        "parameters": {
                            "n_instances": 1
                        },
                        "objects": ["ins", "vpm", "msk"],
                    },
                    {
                        "callback": "RemoveUnseenPoints",
                        "parameters": None,
                        "objects": ["pts", "vpm"],
                    },
                    {
                        "callback": "NormalizePointCords",
                        "parameters": None,
                        "objects": ["pts", "centers"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.CITY_SAMPLE.N_CLASSES,
                        },
                        "objects": ["seg", "proj/seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "rgb",
                            "seg",
                            "msk",
                            "proj/hf",
                            "proj/seg",
                            "proj/tlp",
                            "proj/affmat",
                            "pts",
                        ],
                    },
                ]
            )
        else:
            return utils.transforms.Compose(
                [
                    {
                        "callback": "RandomCrop",
                        "parameters": {
                            "height": cfg.TEST.GAUSSIAN.IMG_CROP_SIZE[1],
                            "width": cfg.TEST.GAUSSIAN.IMG_CROP_SIZE[0],
                            "mode": "center",
                        },
                        "objects": ["rgb", "seg", "vpm", "msk"],
                    },
                    {
                        "callback": "RandomInstance",
                        "parameters": {
                            "n_instances": 1
                        },
                        "objects": ["ins", "vpm", "msk"],
                    },
                    {
                        "callback": "RemoveUnseenPoints",
                        "parameters": None,
                        "objects": ["pts", "vpm"],
                    },
                    {
                        "callback": "NormalizePointCords",
                        "parameters": None,
                        "objects": ["pts", "centers"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.CITY_SAMPLE.N_CLASSES,
                        },
                        "objects": ["seg", "proj/seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "rgb",
                            "seg",
                            "msk",
                            "proj/hf",
                            "proj/seg",
                            "proj/tlp",
                            "proj/affmat",
                            "pts",
                        ],
                    },
                ]
            )
