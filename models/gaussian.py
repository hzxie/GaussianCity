# -*- coding: utf-8 -*-
#
# @File:   gaussian.py
# @Author: Haozhe Xie
# @Date:   2024-01-31 14:11:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-02 15:11:03
# @Email:  root@haozhexie.com
#
# References:
# - https://github.com/graphdeco-inria/gaussian-splatting
# - https://github.dev/VAST-AI-Research/TriplaneGaussian

import math
import numpy as np
import torch
import torch.nn.functional as F


class TruncExp(torch.autograd.Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


class Activations:
    @classmethod
    def inverse_sigmoid(cls, x):
        return np.log(x / (1 - x))

    @classmethod
    def trunc_exp(cls, x):
        return TruncExp.apply(x)


class GSLayer(torch.nn.Module):
    def __init__(self, cfg):
        super(GSLayer, self).__init__()
        self.cfg = cfg
        self.feature_channels = {
            "xyz": 3,
            "opacity": 1,
            "rotation": 4,
            "scaling": 3,
            "shs": 3 if cfg.NETWORK.GAUSSIAN.USE_RGB_ONLY else 48,
        }
        self.layers = torch.nn.ModuleList()
        for name, n_ch in self.feature_channels.items():
            layer = torch.nn.Linear(cfg.NETWORK.GAUSSIAN.FEATURE_DIM, n_ch)
            self._layer_initization(name, layer)
            self.layers.append(layer)

    def _layer_initization(self, name, layer):
        assert name in self.feature_channels.keys(), (
            "Unknown feature in GSLayer: %s" % name
        )
        if name == "xyz":
            torch.nn.init.constant_(layer.weight, 0)
            torch.nn.init.constant_(layer.bias, 0)
        elif name == "opacity":
            torch.nn.init.constant_(
                layer.bias,
                Activations.inverse_sigmoid(self.cfg.NETWORK.GAUSSIAN.INIT_OPACITY),
            )
        elif name == "rotation":
            torch.nn.init.constant_(layer.bias, 0)
            torch.nn.init.constant_(layer.bias[0], 1.0)
        elif name == "scaling":
            torch.nn.init.constant_(layer.bias, self.cfg.NETWORK.GAUSSIAN.INIT_SCALING)
        elif name == "shs" and not self.cfg.NETWORK.GAUSSIAN.USE_RGB_ONLY:
            torch.nn.init.constant_(layer.weight, 0)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, xyz, features):
        gs_attrs = {}
        for name, layer in (self.feature_channels.keys(), self.layers):
            value = layer(features)
            if name == "xyz":
                value = (
                    torch.sigmoid(value) - 0.5
                ) * self.cfg.NETWORK.GAUSSIAN.CLIP_OFFSET
                value = value + xyz
            elif name == "opacity":
                value = torch.sigmoid(value)
            elif name == "rotation":
                return F.normalize(value)
            elif name == "scaling":
                value = Activations.trunc_exp(value)
                if self.cfg.NETWORK.GAUSSIAN.CLIP_SCALING is not None:
                    value = torch.clamp(
                        value, min=0, max=self.cfg.NETWORK.GAUSSIAN.CLIP_SCALING
                    )
            elif name == "shs":
                if self.cfg.NETWORK.GAUSSIAN.USE_RGB_ONLY:
                    value = torch.sigmoid(value)
                value = torch.reshape(value, (value.shape[0], -1, 3))

            gs_attrs[name] = value

        return gs_attrs


class GaussianGenerator(torch.nn.Module):
    def __init__(self, cfg, n_classes):
        super(GaussianGenerator, self).__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.l_xyz = torch.nn.Linear(3, 64)
        self.l_cls = torch.nn.Linear(n_classes, 64)
        self.l_z = torch.nn.Linear(cfg.NETWORK.GAUSSIAN.Z_DIM, 64)
        self.l_f = torch.nn.Linear(64 * 3, cfg.NETWORK.GAUSSIAN.FEATURE_DIM)
        self.te = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=cfg.NETWORK.GAUSSIAN.FEATURE_DIM,
                nhead=cfg.NETWORK.GAUSSIAN.N_ATTENTION_HEADS,
                batch_first=True,
            ),
            num_layers=cfg.NETWORK.GAUSSIAN.N_TRANSFORMER_LAYERS,
        )
        self.l_out = torch.nn.Linear(cfg.NETWORK.GAUSSIAN.FEATURE_DIM, 3)

    def forward(self, points):
        masks = None
        # Create attention masks for padding points
        # n_pts = points.size(1)
        # masks = torch.zeros(
        #     self.cfg.NETWORK.GAUSSIAN.N_ATTENTION_HEADS,
        #     n_pts,
        #     n_pts,
        #     dtype=torch.float32,
        #     device=points.device,
        # )
        # masks[:, :n_act_pts, :n_act_pts] = 1

        f_xyz = self.l_xyz(points[:, :, :3])
        f_cls = self.l_cls(points[:, :, 3 : 3 + self.n_classes])
        f_z = self.l_z(points[:, :, -self.cfg.NETWORK.GAUSSIAN.Z_DIM :])
        f = self.l_f(torch.cat([f_xyz, f_cls, f_z], dim=2))
        f = self.te(f, masks)
        return self.l_out(f)


class GaussianDiscriminator(torch.nn.Module):
    def __init__(self, cfg):
        super(GaussianDiscriminator, self).__init__()
        self.layer = torch.nn.Linear(3, 3)

    def forward(self, points):
        return None
