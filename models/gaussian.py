# -*- coding: utf-8 -*-
#
# @File:   gaussian.py
# @Author: Haozhe Xie
# @Date:   2024-01-31 14:11:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-04 13:12:23
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
    def __init__(self, cfg, n_classes):
        super(GaussianDiscriminator, self).__init__()
        # bottom-up pathway
        # down_conv2d_block = Conv2dBlock, stride=2, kernel=3, padding=1, weight_norm=spectral
        # self.enc1 = down_conv2d_block(num_input_channels, num_filters)  # 3
        self.enc1 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    3,  # RGB
                    cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=2,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.enc2 = down_conv2d_block(1 * num_filters, 2 * num_filters)  # 7
        self.enc2 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    1 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    2 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=2,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.enc3 = down_conv2d_block(2 * num_filters, 4 * num_filters)  # 15
        self.enc3 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    2 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=2,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.enc4 = down_conv2d_block(4 * num_filters, 8 * num_filters)  # 31
        self.enc4 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    8 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=2,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.enc5 = down_conv2d_block(8 * num_filters, 8 * num_filters)  # 63
        self.enc5 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    8 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    8 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=2,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # top-down pathway
        # latent_conv2d_block = Conv2dBlock, stride=1, kernel=1, weight_norm=spectral
        # self.lat2 = latent_conv2d_block(2 * num_filters, 4 * num_filters)
        self.lat2 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    2 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=1,
                    kernel_size=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.lat3 = latent_conv2d_block(4 * num_filters, 4 * num_filters)
        self.lat3 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=1,
                    kernel_size=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.lat4 = latent_conv2d_block(8 * num_filters, 4 * num_filters)
        self.lat4 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    8 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=1,
                    kernel_size=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.lat5 = latent_conv2d_block(8 * num_filters, 4 * num_filters)
        self.lat5 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    8 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=1,
                    kernel_size=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # upsampling
        self.upsample2x = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # final layers
        # stride1_conv2d_block = Conv2dBlock, stride=1, kernel=3, padding=1, weight_norm=spectral
        # self.final2 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
        self.final2 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    4 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    2 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                    stride=1,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        )
        # self.output = Conv2dBlock(num_filters * 2, num_labels + 1, kernel_size=1)
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(
                2 * cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE,
                n_classes + 1,
                stride=1,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.LeakyReLU(0.2),
        )
        self.interpolator = self._smooth_interp

    @staticmethod
    def _smooth_interp(x, size):
        r"""Smooth interpolation of segmentation maps.

        Args:
            x (4D tensor): Segmentation maps.
            size(2D list): Target size (H, W).
        """
        x = F.interpolate(x, size=size, mode="area")
        onehot_idx = torch.argmax(x, dim=-3, keepdims=True)
        x.fill_(0.0)
        x.scatter_(1, onehot_idx, 1.0)
        return x

    def forward(self, images, seg_maps, masks):
        # bottom-up pathway
        feat11 = self.enc1(images * masks)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.upsample2x(feat25) + self.lat4(feat14)
        feat23 = self.upsample2x(feat24) + self.lat3(feat13)
        feat22 = self.upsample2x(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)

        label_map = self.interpolator(seg_maps * masks, size=feat32.size()[2:])
        pred = self.output(feat32)  # N, num_labels + 1, H//4, W//4

        return {"pred": pred, "label": label_map}
