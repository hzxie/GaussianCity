# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Haozhe Xie
# @Date:   2023-12-13 13:40:10
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-12-13 15:45:57
# @Email:  root@haozhexie.com

import torch

import topdown_projector_ext


class TopDownProjector(torch.nn.Module):
    def __init__(self):
        super(TopDownProjector, self).__init__()

    def forward(self, volume):
        return TopDownProjectorFunction.apply(volume)


class TopDownProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume):
        ctx.save_for_backward(volume.size())
        return topdown_projector_ext.forward(volume)

    @staticmethod
    def backward(ctx, grad_height_field, grad_seg_map):
        b, c, h, w, d = ctx.saved_tensors
        return torch.ones(
            [b, c, h, w, d],
            dtype=grad_height_field.dtype,
            device=grad_height_field.device,
        )
