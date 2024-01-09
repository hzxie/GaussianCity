# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Haozhe Xie
# @Date:   2023-12-23 11:30:15
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-09 15:04:10
# @Email:  root@haozhexie.com

import torch

import footprint_extruder_ext


class FootprintExtruder(torch.nn.Module):
    def __init__(
        self,
        l1_height=0,
        roof_height=1,
        l1_id_offset=0,
        roof_id_offset=1,
        footprint_id_range=[100, 5000],
        max_height=384,
    ):
        super(FootprintExtruder, self).__init__()
        self.l1_height = l1_height
        self.roof_height = roof_height
        self.l1_id_offset = l1_id_offset
        self.roof_id_offset = roof_id_offset
        self.footprint_id_range = footprint_id_range
        self.max_height = max_height

    def forward(self, height_field, seg_map):
        assert torch.max(height_field) < self.max_height, "Max Value %d" % torch.max(
            height_field
        )
        return FootprintExtruderFunction.apply(
            height_field,
            seg_map,
            self.l1_height,
            self.roof_height,
            self.l1_id_offset,
            self.roof_id_offset,
            self.footprint_id_range,
            self.max_height,
        )


class FootprintExtruderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        height_field,
        seg_map,
        l1_height,
        roof_height,
        l1_id_offset,
        roof_id_offset,
        footprint_id_range,
        max_height,
    ):
        # height_field.shape: (B, C, H, W)
        # seg_map.shape: (B, C, H, W)
        return footprint_extruder_ext.forward(
            height_field,
            seg_map,
            l1_height,
            roof_height,
            l1_id_offset,
            roof_id_offset,
            footprint_id_range[0],
            footprint_id_range[1],
            max_height,
        )
