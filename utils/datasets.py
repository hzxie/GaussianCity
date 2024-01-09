# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-09 10:44:32
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
    elif dataset_name == "CITY_SAMPLE_BUILDING":
        return CitySampleBuildingDataset(cfg, split)
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
        self.fields = ["hf", "seg", "footage", "raycasting"]
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
        rendering = self.renderings[idx % self.n_renderings]
        data = {
            "hf": self._get_height_field(rendering["hf"], self.cfg),
            "seg": self._get_seg_layout(rendering["seg"]),
            "footage": self._get_footage_img(rendering["footage"]),
        }
        raycasting = utils.io.IO.get(rendering["raycasting"])
        data["voxel_id"] = raycasting["voxel_id"]
        data["depth2"] = raycasting["depth2"]
        data["raydirs"] = raycasting["raydirs"]
        data["cam_origin"] = raycasting["cam_origin"]
        data["mask"] = raycasting["mask"]
        data = self.transforms(data)
        return data

    def _get_renderings(self, cfg, split):
        cities = [
            "City%02d" % (i + 1) for i in range(cfg.DATASETS.CITY_SAMPLE.N_CITIES)
        ]
        files = [
            {
                "name": "%s/%s/%04d" % (c, s, i),
                "hf": os.path.join(cfg.DATASETS.CITY_SAMPLE.DIR, c, "HeightField.png"),
                "seg": os.path.join(cfg.DATASETS.CITY_SAMPLE.DIR, c, "SegLayout.png"),
                "footage": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR,
                    c,
                    "ColorImage",
                    s,
                    "%sSequence.%04d.jpeg" % (c, i),
                ),
                "raycasting": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR, c, "Raycasting", "%04d.pkl" % i
                ),
                "footprint_bboxes": os.path.join(
                    cfg.DATASETS.CITY_SAMPLE.DIR, c, "Footprints.pkl"
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
                elif k == "hf":
                    self.memcached[v] = self._get_height_field(v, cfg)
                elif k == "seg":
                    self.memcached[v] = self._get_seg_layout(v)

        return files if split == "train" else files[-32:]

    def _get_height_field(self, file_path, cfg):
        return (
            np.array(utils.io.IO.get(file_path)) / cfg.DATASETS.CITY_SAMPLE.MAX_HEIGHT
        )

    def _get_seg_layout(self, file_path):
        if file_path in self.memcached:
            return self.memcached[file_path]

        return np.array(utils.io.IO.get(file_path).convert("P"))

    def _get_footage_img(self, file_path):
        img = utils.io.IO.get(file_path)
        return (np.array(img) / 255.0 - 0.5) * 2

    def _get_data_transforms(self, cfg, split):
        BULIDING_MASK_ID = 2
        if split == "train":
            return utils.transforms.Compose(
                [
                    {
                        "callback": "RandomCrop",
                        "parameters": {
                            "height": cfg.TRAIN.GANCRAFT.CROP_SIZE[1],
                            "width": cfg.TRAIN.GANCRAFT.CROP_SIZE[0],
                        },
                        "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
                    },
                    {
                        "callback": "BuildingMaskRemap",
                        "parameters": {
                            "bld_facade_label": BULIDING_MASK_ID,
                            "bld_roof_label": BULIDING_MASK_ID,
                            "min_bld_ins_id": 10,
                        },
                        "objects": ["voxel_id", "seg"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.CITY_SAMPLE.N_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "hf",
                            "seg",
                            "voxel_id",
                            "depth2",
                            "raydirs",
                            "cam_origin",
                            "footage",
                            "mask",
                        ],
                    },
                ]
            )
        else:
            return utils.transforms.Compose(
                [
                    {
                        "callback": "CenterCrop",
                        "parameters": {
                            "height": cfg.TEST.GANCRAFT.CROP_SIZE[1],
                            "width": cfg.TEST.GANCRAFT.CROP_SIZE[0],
                        },
                        "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
                    },
                    {
                        "callback": "BuildingMaskRemap",
                        "parameters": {
                            "bld_facade_label": BULIDING_MASK_ID,
                            "bld_roof_label": BULIDING_MASK_ID,
                            "min_bld_ins_id": 10,
                        },
                        "objects": ["voxel_id", "seg"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.CITY_SAMPLE.N_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "hf",
                            "seg",
                            "voxel_id",
                            "depth2",
                            "raydirs",
                            "cam_origin",
                            "footage",
                            "mask",
                        ],
                    },
                ]
            )


class CitySampleBuildingDataset(CitySampleDataset):
    def __init__(self, cfg, split):
        super(CitySampleBuildingDataset, self).__init__(cfg, split)
        self.split = split
        # Overwrite the transforms in CitySampleDataset
        self.transforms = self._get_data_transforms(cfg, split)

    def __len__(self):
        return (
            self.n_renderings * self.cfg.DATASETS.CITY_SAMPLE_BUILDING.N_REPEAT
            if self.split == "train"
            else self.n_renderings
        )

    def __getitem__(self, idx):
        data = None
        while data is None:
            rendering = self.renderings[idx % self.n_renderings]
            data = self._get_data(rendering)
            idx += 1

        return data

    def _get_data(self, rendering):
        data = {"footage": self._get_footage_img(rendering["footage"])}
        raycasting = utils.io.IO.get(rendering["raycasting"])
        footprint_bboxes = utils.io.IO.get(rendering["footprint_bboxes"])
        data["voxel_id"] = raycasting["voxel_id"]
        data["depth2"] = raycasting["depth2"]
        data["raydirs"] = raycasting["raydirs"]
        data["cam_origin"] = raycasting["cam_origin"]
        data["mask"] = raycasting["mask"]
        # Determine Building Instances
        data["building_id"] = self._get_rnd_building_id(
            data["voxel_id"][..., 0, 0],
            data["mask"],
            True if self.split == "train" else False,
        )
        # Cannot find suitable buildings in the current view
        if data["building_id"] is None:
            return None

        # NOTE: data["footprint_bboxes"] -> (dy, dx, h, w)
        data["footprint_bboxes"] = self._get_building_stats(
            footprint_bboxes, data["building_id"]
        )
        data["hf"] = self._get_hf_seg(
            "hf",
            rendering,
            self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE + int(data["footprint_bboxes"][1]),
            self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE + int(data["footprint_bboxes"][0]),
        )
        data["seg"] = self._get_hf_seg(
            "seg",
            rendering,
            self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE + int(data["footprint_bboxes"][1]),
            self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE + int(data["footprint_bboxes"][0]),
        )
        data = self.transforms(data)
        return data

    def _get_hf_seg(self, field, rendering, cx, cy):
        if field in self.cfg.DATASETS.CITY_SAMPLE_BUILDING.PIN_MEMORY:
            img = self.memcached[rendering[field]]
        elif field == "hf":
            img = self._get_height_field(rendering[field], self.cfg)
        elif field == "seg":
            img = self._get_seg_layout(rendering[field])
        else:
            raise Exception("Unknown field: %s" % field)

        size = self.cfg.DATASETS.CITY_SAMPLE_BUILDING.VOL_SIZE
        half_size = size // 2
        pad_img = (
            np.zeros((size, size))
            if len(img.shape) == 2
            else np.zeros((size, size, img.shape[2]))
        )
        # Determine the crop position
        tl_x, br_x = cx - half_size, cx + half_size
        tl_y, br_y = cy - half_size, cy + half_size
        # Handle Corner case (out of bounds)
        pad_x = 0 if tl_x >= 0 else abs(tl_x)
        tl_x = tl_x if tl_x >= 0 else 0
        br_x = min(br_x, self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE)
        patch_w = br_x - tl_x
        pad_y = 0 if tl_y >= 0 else abs(tl_y)
        tl_y = tl_y if tl_y >= 0 else 0
        br_y = min(br_y, self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE)
        patch_h = br_y - tl_y
        # Copy-paste
        pad_img[pad_y : pad_y + patch_h, pad_x : pad_x + patch_w] = img[
            tl_y:br_y, tl_x:br_x
        ]
        return pad_img

    def _get_rnd_building_id(self, voxel_id, seg_mask, rnd_mode=True, n_max_times=100):
        BLD_INS_LABEL_MIN = 100
        N_MIN_PIXELS = 64

        buliding_ids = np.unique(voxel_id[voxel_id >= BLD_INS_LABEL_MIN])
        # NOTE: The facade instance IDs are multiple of 4.
        buliding_ids = buliding_ids[buliding_ids % 4 == 0]
        # Fix bld_idx in test mode
        n_bulidings = len(buliding_ids)
        # Fix a bug causes empty range for randrange() (0, 0, 0) for random.randint()
        if n_bulidings == 0:
            return None

        bld_idx = n_bulidings // 4
        # Make sure that the building contains unambiguous pixels
        n_times = 0
        while n_times < n_max_times:
            n_times += 1
            if rnd_mode:
                bld_idx = random.randint(0, n_bulidings - 1)
            else:
                bld_idx += 1

            building_id = buliding_ids[bld_idx % n_bulidings]
            if np.count_nonzero(seg_mask[voxel_id == building_id]) >= N_MIN_PIXELS:
                break

        assert building_id % 4 == 0, "Building instance ID MUST BE an even number."
        return building_id if n_times < n_max_times else None

    def _get_footprint_bboxes(self, footprint_bboxes, building_id):
        BLD_INS_LABEL_MIN = 100
        assert building_id >= BLD_INS_LABEL_MIN
        # NOTE: 0 <= dx, dy < 1536, indicating the offsets between the building
        # and the image center.
        x, y, w, h = footprint_bboxes[building_id]
        # See also: https://git.haozhexie.com/hzxie/city-dreamer/src/branch/master/scripts/dataset_generator.py#L503-L511
        dx = x - self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE // 2 + w / 2
        dy = y - self.cfg.DATASETS.CITY_SAMPLE.VOL_SIZE // 2 + h / 2
        return torch.Tensor([dy, dx, h, w, building_id])

    def _get_data_transforms(self, cfg, split):
        BULIDING_FACADE_ID = 6
        BULIDING_ROOF_ID = 7
        if split == "train":
            return utils.transforms.Compose(
                [
                    {
                        "callback": "BuildingMaskRemap",
                        "parameters": {
                            "attr": "building_id",
                            "bld_facade_label": BULIDING_FACADE_ID,
                            "bld_roof_label": BULIDING_ROOF_ID,
                            "min_bld_ins_id": 10,
                        },
                        "objects": ["voxel_id", "seg"],
                    },
                    {
                        "callback": "MaskRaydirs",
                        "parameters": {
                            "attr": "raydirs",
                            "values": [BULIDING_FACADE_ID, BULIDING_ROOF_ID],
                        },
                    },
                    {
                        "callback": "CenterCropTarget",
                        "parameters": {
                            "height": cfg.TRAIN.GANCRAFT.CROP_SIZE[1],
                            "width": cfg.TRAIN.GANCRAFT.CROP_SIZE[0],
                            "target_value": BULIDING_FACADE_ID,
                        },
                        "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.CITY_SAMPLE.N_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "hf",
                            "seg",
                            "voxel_id",
                            "depth2",
                            "raydirs",
                            "cam_origin",
                            "footage",
                            "mask",
                        ],
                    },
                ]
            )
        else:
            return utils.transforms.Compose(
                [
                    {
                        "callback": "BuildingMaskRemap",
                        "parameters": {
                            "attr": "building_id",
                            "bld_facade_label": BULIDING_FACADE_ID,
                            "bld_roof_label": BULIDING_ROOF_ID,
                            "min_bld_ins_id": 10,
                        },
                        "objects": ["voxel_id", "seg"],
                    },
                    {
                        "callback": "MaskRaydirs",
                        "parameters": {
                            "attr": "raydirs",
                            "values": [BULIDING_FACADE_ID, BULIDING_ROOF_ID],
                        },
                    },
                    {
                        "callback": "CenterCropTarget",
                        "parameters": {
                            "height": cfg.TRAIN.GANCRAFT.CROP_SIZE[1],
                            "width": cfg.TRAIN.GANCRAFT.CROP_SIZE[0],
                            "target_value": BULIDING_FACADE_ID,
                        },
                        "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.CITY_SAMPLE.N_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "hf",
                            "seg",
                            "voxel_id",
                            "depth2",
                            "raydirs",
                            "cam_origin",
                            "footage",
                            "mask",
                        ],
                    },
                ]
            )
