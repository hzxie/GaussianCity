# -*- coding: utf-8 -*-
#
# @File:   discriminator.py
# @Author: Haozhe Xie
# @Date:   2024-03-09 20:37:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-12 18:57:09
# @Email:  root@haozhexie.com

import torch
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, cfg, n_classes):
        super(Discriminator, self).__init__()
        self.cfg = cfg
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

    def _single_forward(self, images, seg_maps):
        # bottom-up pathway
        feat11 = self.enc1(images)
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

        label_map = self.interpolator(seg_maps, size=feat32.size()[2:])
        pred = self.output(feat32)  # N, num_labels + 1, H//4, W//4
        return {"pred": pred, "label": label_map}

    def forward(self, images, seg_maps, masks):
        # print(seg_maps.size())  # torch.Size([1, 7, H, W])
        # print(masks.size())  # torch.Size([1, 1, H, W])
        return self._single_forward(images * masks, seg_maps * masks)
