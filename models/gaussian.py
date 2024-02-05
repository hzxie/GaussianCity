# -*- coding: utf-8 -*-
#
# @File:   gaussian.py
# @Author: Haozhe Xie
# @Date:   2024-01-31 14:11:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-05 16:51:04
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


class Camera:
    def __init__(self, k, w2c, height, width, z_near=0.01, z_far=100.0):
        self.height = height
        self.width = width
        self.z_near = z_near
        self.z_far = z_far

        self.fov_x, self.fov_y = self._get_fov_from_intrinsics(k, height, width)
        self.world_view_transform = np.transpose(w2c, (0, 1))
        self.projection_matrix = self._get_projection_matrix(
            self.z_near, self.z_far, self.fov_x, self.fov_y
        ).to(w2c.device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def get_camera_from_c2w(k, c2w, height, width):
        return Camera(k, torch.inverse(c2w), height, width)

    @staticmethod
    def _get_projection_matrix(z_near, z_far, fov_x, fov_y):
        tan_half_fov_y = math.tan((fov_y / 2))
        tan_half_fov_x = math.tan((fov_x / 2))
        top = tan_half_fov_y * z_near
        bottom = -top
        right = tan_half_fov_x * z_far
        left = -right

        P = torch.zeros(4, 4)
        z_sign = 1.0
        P[0, 0] = 2.0 * z_near / (right - left)
        P[1, 1] = 2.0 * z_near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * z_far / (z_far - z_near)
        P[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return P.transpose(0, 1)

    @staticmethod
    def _get_fov_from_intrinsics(k, w, h):
        fx, fy = k[0, 0], k[1, 1]
        fov_x = 2 * torch.arctan2(w, 2 * fx)
        fov_y = 2 * torch.arctan2(h, 2 * fy)
        return fov_x, fov_y


class GaussianModel:
    def __init__(self, xyz, opacity, rotation, scaling, shs):
        self.xyz = xyz
        self.opacity = opacity
        self.rotation = rotation
        self.scaling = scaling
        self.shs = shs

    def get_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            attrs.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            attrs.append("f_rest_{}".format(i))

        attrs.append("opacity")
        for i in range(self.scaling.shape[1]):
            attrs.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            attrs.append("rot_{}".format(i))

        return attrs

    def get_ply_elements(self):
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = Activations.inverse_sigmoid(
            torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().numpy()
        )
        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.get_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        return elements


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

        return GaussianModel(**gs_attrs)


class GSRenderer(torch.nn.Module):
    def __init__(self, cfg):
        super(GSRenderer, self).__init__()
