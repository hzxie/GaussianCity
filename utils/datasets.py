# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-05-02 21:37:11
# @Email:  root@haozhexie.com

import copy
import json
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
        return GoogleDataset(cfg, split)
    elif dataset_name == "KITTI_360":
        return Kitti360Dataset(cfg, split)
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.dataset_cfg = cfg
        self.split = split
        self.memcached = {}
        self.transforms = self._get_data_transforms(cfg, split)
        # Dummy parameters to be filled by the inherited classes
        self.renderings = []
        self.n_renderings = 0

    def get_K(self):
        return np.array(self.dataset_cfg.CAM_K, dtype=np.float32).reshape((3, 3))

    def get_sensor_size(self):
        return self.dataset_cfg.SENSOR_SIZE

    def is_flip_ud(self):
        return self.dataset_cfg.FLIP_UD

    def get_n_classes(self):
        return self.dataset_cfg.N_CLASSES

    def get_special_z_scale_classes(self):
        return list(self.dataset_cfg.Z_SCALE_SPECIAL_CLASSES.values())

    def get_proj_size(self):
        return self.dataset_cfg.PROJ_SIZE

    def pin_memory(self, files, keys=[]):
        for f in tqdm(files, desc="Loading partial files to RAM"):
            for k, v in f.items():
                if k not in keys:
                    continue
                elif v in self.memcached:
                    continue
                else:
                    self.memcached[v] = utils.io.IO.get(v)

    def __len__(self):
        return (
            self.n_renderings * self.dataset_cfg.N_REPEAT
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
            / self.dataset_cfg.SCALE
        )
        cam_pos[:2] += self.dataset_cfg.MAP_SIZE // 2

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
            "vpm": pts["vpm"],
            "msk": pts["msk"],
            "pts": pts["pts"],
        }
        if "affmat" in pts["prj"] and "tlp" in pts["prj"]:
            data["proj/affmat"] = pts["prj"]["affmat"]
            data["proj/tlp"] = pts["prj"]["tlp"]

        data = self.transforms(data)
        return data

    def _get_data_transforms(self, cfg, split):
        if split == "train":
            return utils.transforms.Compose(
                [
                    {
                        "callback": "RandomCrop",
                        "parameters": {
                            "height": cfg.TRAIN_CROP_SIZE[1],
                            "width": cfg.TRAIN_CROP_SIZE[0],
                            "n_min_pixels": cfg.TRAIN_MIN_PIXELS,
                            "n_max_points": cfg.TRAIN_MAX_POINTS,
                        },
                        "objects": ["rgb", "seg", "ins", "vpm", "msk"],
                    },
                    # {
                    #     "callback": "RandomInstance",
                    #     "parameters": {
                    #         "n_instances": 1
                    #     },
                    #     "objects": ["ins", "vpm", "msk"],
                    # },
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
                            "n_classes": self.get_n_classes(),
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
                            "height": cfg.TEST_CROP_SIZE[1],
                            "width": cfg.TEST_CROP_SIZE[0],
                            "mode": "center",
                        },
                        "objects": ["rgb", "seg", "ins", "vpm", "msk"],
                    },
                    # {
                    #     "callback": "RandomInstance",
                    #     "parameters": {
                    #         "n_instances": 1
                    #     },
                    #     "objects": ["ins", "vpm", "msk"],
                    # },
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
                            "n_classes": self.get_n_classes(),
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


class GoogleDataset(Dataset):
    def __init__(self, cfg, split):
        super(GoogleDataset, self).__init__(cfg.DATASETS.GOOGLE_EARTH, split)
        self.cfg = cfg
        self.renderings = self._get_renderings(cfg.DATASETS.GOOGLE_EARTH, split)
        self.n_renderings = len(self.renderings)

    def instances_to_classes(self, instances):
        # Make it compatible in both numpy and PyTorch
        cfg = self.cfg.DATASETS.GOOGLE_EARTH
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

        classes = copy.deepcopy(instances)
        classes[bldg_facade_idx] = cfg.BLDG_FACADE_CLSID
        classes[bldg_roof_idx] = cfg.BLDG_ROOF_CLSID
        return classes

    def _get_renderings(self, cfg, split):
        cities = sorted(os.listdir(cfg.DIR))[: cfg.N_CITIES]
        files = [
            {
                "name": "%s/%02d" % (c, i),
                # Camera parameters
                "Rt": os.path.join(cfg.DIR, c, "CameraPoses.csv"),
                # The XY centers of the instances
                "centers": os.path.join(cfg.DIR, c, "CENTERS.pkl"),
                "rgb": os.path.join(
                    cfg.DIR,
                    c,
                    "footage",
                    "%s_%02d.jpeg" % (c, i),
                ),
                "seg": os.path.join(
                    cfg.DIR,
                    c,
                    "seg",
                    "%s_%02d.png" % (c, i),
                ),
                "ins": os.path.join(
                    cfg.DIR,
                    c,
                    "InstanceImage",
                    "%04d.png" % i,
                ),
                # Projection
                "proj/hf": os.path.join(cfg.DIR, c, "Projection", "REST-TD_HF.png"),
                "proj/seg": os.path.join(cfg.DIR, c, "Projection", "REST-SEG.png"),
                # Precomputed Points in the viewpoint (scripts/dataset_generator.py)
                "pts": os.path.join(cfg.DIR, c, "Points", "%04d.pkl" % i),
            }
            for c in cities
            for i in range(cfg.N_VIEWS)
        ]
        if cfg.PIN_MEMORY:
            self.pin_memory(files, cfg.PIN_MEMORY)

        return (
            files
            if split == "train"
            else [f for f in files if f["name"].endswith("00")]
        )


class Kitti360Dataset(Dataset):
    def __init__(self, cfg, split):
        super(Kitti360Dataset, self).__init__(cfg.DATASETS.KITTI_360, split)
        self.cfg = cfg
        self.renderings = self._get_renderings(cfg.DATASETS.KITTI_360, split)
        self.n_renderings = len(self.renderings)

    def instances_to_classes(self, instances):
        # Make it compatible in both numpy and PyTorch
        cfg = self.cfg.DATASETS.KITTI_360
        bldg_facade_idx = (instances >= cfg.BLDG_RANGE[0]) & (
            instances < cfg.BLDG_RANGE[1]
        )
        car_idx = (instances >= cfg.CAR_RANGE[0]) & (instances < cfg.CAR_RANGE[1])

        classes = copy.deepcopy(instances)
        classes[bldg_facade_idx] = cfg.BLDG_FACADE_CLSID
        classes[car_idx] = cfg.CAR_CLSID
        return classes

    def _get_renderings(self, cfg, split):
        if os.path.exists(cfg.VIEW_INDEX_FILE):
            with open(cfg.VIEW_INDEX_FILE, "r") as fp:
                view_idx = json.load(fp)
        else:
            view_idx = {}
            cities = sorted(os.listdir(cfg.DIR))
            for c in cities:
                pts_dir = os.path.join(cfg.DIR, c, "Points")
                if os.path.exists(pts_dir):
                    view_idx[c] = [int(f[:-4]) for f in sorted(os.listdir(pts_dir))]
            with open(cfg.VIEW_INDEX_FILE, "w") as fp:
                json.dump(view_idx, fp, indent=2)

        files = [
            {
                "name": "%s/%010d" % (c, i),
                # Camera parameters
                "Rt": os.path.join(cfg.DIR, c, "CameraPoses.csv"),
                # The XY centers of the instances
                "centers": os.path.join(cfg.DIR, c, "CENTERS.pkl"),
                "rgb": os.path.join(
                    cfg.DIR,
                    c,
                    "footage",
                    "%010d.png" % i,
                ),
                "seg": os.path.join(
                    cfg.DIR,
                    c,
                    "seg",
                    "%010d.png" % i,
                ),
                "ins": os.path.join(
                    cfg.DIR,
                    c,
                    "InstanceImage",
                    "%010d.png" % i,
                ),
                # Projection
                "proj/hf": os.path.join(cfg.DIR, c, "Projection", "REST-TD_HF.png"),
                "proj/seg": os.path.join(cfg.DIR, c, "Projection", "REST-SEG.png"),
                # Precomputed Points in the viewpoint (scripts/dataset_generator.py)
                "pts": os.path.join(cfg.DIR, c, "Points", "%010d.pkl" % i),
            }
            for c, v in view_idx.items()
            for i in v
        ]
        if cfg.PIN_MEMORY:
            self.pin_memory(files, cfg.PIN_MEMORY)

        return (
            files
            if split == "train"
            else [f for i, f in enumerate(files) if i % 1000 == 0]
        )


class CitySampleDataset(Dataset):
    def __init__(self, cfg, split):
        super(CitySampleDataset, self).__init__(cfg.DATASETS.CITY_SAMPLE, split)
        self.cfg = cfg
        self.renderings = self._get_renderings(cfg.DATASETS.CITY_SAMPLE, split)
        self.n_renderings = len(self.renderings)

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

    def _get_renderings(self, cfg, split):
        cities = ["City%02d" % (i + 1) for i in range(cfg.N_CITIES)]
        files = [
            {
                "name": "%s/%s/%04d" % (c, s, i),
                # Camera parameters
                "Rt": os.path.join(cfg.DIR, c, "CameraPoses.csv"),
                # The XY centers of the instances
                "centers": os.path.join(cfg.DIR, c, "CENTERS.pkl"),
                "rgb": os.path.join(
                    cfg.DIR,
                    c,
                    "ColorImage",
                    s,
                    "%sSequence.%04d.jpeg" % (c, i),
                ),
                "seg": os.path.join(
                    cfg.DIR,
                    c,
                    "SemanticImage",
                    "%sSequence.%04d.png" % (c, i),
                ),
                "ins": os.path.join(
                    cfg.DIR,
                    c,
                    "InstanceImage",
                    "%04d.png" % i,
                ),
                # Projection
                "proj/hf": os.path.join(cfg.DIR, c, "Projection", "REST-TD_HF.png"),
                "proj/seg": os.path.join(cfg.DIR, c, "Projection", "REST-SEG.png"),
                # Precomputed Points in the viewpoint (scripts/dataset_generator.py)
                "pts": os.path.join(cfg.DIR, c, "Points", "%04d.pkl" % i),
            }
            for c in cities
            for i in range(cfg.N_VIEWS)
            for s in cfg.CITY_STYLES
        ]
        if cfg.PIN_MEMORY:
            self.pin_memory(files, cfg.PIN_MEMORY)

        return (
            files
            if split == "train"
            else [
                f
                for f in files
                if f["name"].endswith("000") or f["name"].endswith("500")
            ]
        )
