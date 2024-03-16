# -*- coding: utf-8 -*-
#
# @File:   generator.py
# @Author: Haozhe Xie
# @Date:   2024-03-09 20:36:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-17 07:08:57
# @Email:  root@haozhexie.com

import numpy as np
import torch
import torch.nn.functional as F


class Generator(torch.nn.Module):
    def __init__(self, cfg, n_classes):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.encoder = ProjectionEncoder(
            n_classes, cfg.NETWORK.GAUSSIAN.PROJ_ENCODER_OUT_DIM - 4
        )
        self.pos_encoder = SinCosEncoder(cfg.NETWORK.GAUSSIAN.N_FREQ_BANDS)
        self.ga_mlp = GaussianAttrMLP(
            n_classes,
            2
            * cfg.NETWORK.GAUSSIAN.PROJ_ENCODER_OUT_DIM
            * cfg.NETWORK.GAUSSIAN.N_FREQ_BANDS,
            cfg.NETWORK.GAUSSIAN.Z_DIM,
            cfg.NETWORK.GAUSSIAN.MLP_HIDDEN_DIM,
            cfg.NETWORK.GAUSSIAN.ATTR_FACTORS,
            cfg.NETWORK.GAUSSIAN.ATTR_N_LAYERS,
        )

    def forward(self, proj_uv, rel_xyz, onehots, z, dpt_pe, proj_hf, proj_seg):
        proj_feat = self.encoder(proj_hf, proj_seg)
        pt_feat = (
            F.grid_sample(proj_feat, proj_uv.unsqueeze(dim=1), align_corners=True)
            .squeeze(dim=2)
            .permute(0, 2, 1)
        )
        pt_feat = torch.cat([pt_feat, rel_xyz, dpt_pe], dim=2)
        pt_feat = self.pos_encoder(pt_feat)
        # print(pt_feat.size())   # torch.Size([bs, n_pts, 1024]
        return self.ga_mlp(pt_feat, onehots, z)


class ProjectionEncoder(torch.nn.Module):
    def __init__(self, n_classes, out_channels):
        super(ProjectionEncoder, self).__init__()
        self.hf_conv = torch.nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.seg_conv = torch.nn.Conv2d(
            n_classes, 32, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = torch.nn.GroupNorm(32, 64)
        self.conv2 = ResConvBlock(64, 128)
        self.conv3 = ResConvBlock(128, 256)
        self.conv4 = ResConvBlock(256, 512)
        self.dconv5 = torch.nn.ConvTranspose2d(
            512, 128, kernel_size=4, stride=2, padding=1
        )
        self.dconv6 = torch.nn.ConvTranspose2d(
            128, 32, kernel_size=4, stride=2, padding=1
        )
        self.dconv7 = torch.nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, proj_hf, proj_seg):
        hf = self.hf_conv(proj_hf)
        seg = self.seg_conv(proj_seg)
        out = F.relu(self.bn1(torch.cat([hf, seg], dim=1)), inplace=True)
        # print(out.size())   # torch.Size([N, 64, H/2, W/2])
        out = F.avg_pool2d(self.conv2(out), 2, stride=2)
        # print(out.size())   # torch.Size([N, 128, H/4, W/4])
        out = self.conv3(out)
        # print(out.size())   # torch.Size([N, 256, H/4, W/4])
        out = self.conv4(out)
        # print(out.size())   # torch.Size([N, 512, H/4, W/4])
        out = self.dconv5(out)
        # print(out.size())   # torch.Size([N, 128, H/2, W/2])
        out = self.dconv6(out)
        # print(out.size())   # torch.Size([N, 32, H, W])
        out = self.dconv7(out)
        # print(out.size())   # torch.Size([N, OUT_DIM - 1, H, W])
        return torch.tanh(out)


class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResConvBlock, self).__init__()
        # conv3x3(in_planes, int(out_planes / 2))
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        # conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        # conv3x3(int(out_planes / 4), int(out_planes / 4))
        self.conv3 = torch.nn.Conv2d(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.bn1 = torch.nn.GroupNorm(32, in_channels)
        self.bn2 = torch.nn.GroupNorm(32, out_channels // 2)
        self.bn3 = torch.nn.GroupNorm(32, out_channels // 4)
        self.bn4 = torch.nn.GroupNorm(32, in_channels)

        if in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                self.bn4,
                torch.nn.ReLU(True),
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        # print(residual.size())      # torch.Size([N, 64, H, W])
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        # print(out1.size())          # torch.Size([N, 64, H, W])
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        # print(out2.size())          # torch.Size([N, 32, H, W])
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        # print(out3.size())          # torch.Size([N, 32, H, W])
        out3 = torch.cat((out1, out2, out3), dim=1)
        # print(out3.size())          # torch.Size([N, 128, H, W])
        if self.downsample is not None:
            residual = self.downsample(residual)
            # print(residual.size())  # torch.Size([N, 128, H, W])
        out3 += residual
        return out3


class SinCosEncoder(torch.nn.Module):
    def __init__(self, n_freq_bands=8):
        super(SinCosEncoder, self).__init__()
        self.freq_bands = 2.0 ** torch.linspace(
            0,
            n_freq_bands - 1,
            steps=n_freq_bands,
        )

    def forward(self, features):
        cord_sin = torch.cat(
            [torch.sin(features * fb) for fb in self.freq_bands], dim=-1
        )
        cord_cos = torch.cat(
            [torch.cos(features * fb) for fb in self.freq_bands], dim=-1
        )
        return torch.cat([cord_sin, cord_cos], dim=-1)


class GaussianAttrMLP(torch.nn.Module):
    r"""MLP with affine modulation."""

    def __init__(self, n_classes, in_dim, z_dim, hidden_dim, factors={}, n_layers={}):
        super(GaussianAttrMLP, self).__init__()
        self.factors = factors
        self.n_layers = n_layers
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)
        self.fc_m_a = torch.nn.Linear(
            n_classes,
            hidden_dim,
            bias=False,
        )
        self.fc_1 = torch.nn.Linear(
            in_dim,
            hidden_dim,
        )
        self.fc_2 = ModLinear(
            hidden_dim,
            hidden_dim,
            z_dim,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_3 = ModLinear(
            hidden_dim,
            hidden_dim,
            z_dim,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_4 = ModLinear(
            hidden_dim,
            hidden_dim,
            z_dim,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_5 = ModLinear(
            hidden_dim,
            hidden_dim,
            z_dim,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        for k in factors.keys():
            assert k in ["xyz", "rgb", "scale", "opacity"], "Unknwon key: %s" % k
            for i in range(n_layers[k]):
                setattr(
                    self,
                    "fc_6_%s_%d" % (k, i),
                    ModLinear(
                        hidden_dim,
                        hidden_dim,
                        z_dim,
                        bias=False,
                        mod_bias=True,
                        output_mode=True,
                    ),
                )
            setattr(
                self,
                "fc_out_%s" % k,
                torch.nn.Linear(
                    hidden_dim,
                    1 if k == "opacity" else 3,
                ),
            )

    def forward(self, pt_feat, onehots, z):
        f = self.fc_1(pt_feat)
        f = f + self.fc_m_a(onehots)

        f = self.act(f)
        f = self.act(self.fc_2(f, z))
        f = self.act(self.fc_3(f, z))
        f = self.act(self.fc_4(f, z))
        f = self.act(self.fc_5(f, z))
        output = {}
        for k in self.factors.keys():
            _f = f.clone()
            for i in range(self.n_layers[k]):
                fc_6 = getattr(self, "fc_6_%s_%d" % (k, i))
                _f = self.act(fc_6(_f, z))

            fc_out = getattr(self, "fc_out_%s" % k)
            output[k] = fc_out(_f)

        if "xyz" in self.factors:
            output["xyz"] = (torch.sigmoid(output["xyz"]) - 0.5) * self.factors["xyz"]
        if "rgb" in self.factors:
            output["rgb"] = output["rgb"] * self.factors["rgb"]
        if "scale" in self.factors:
            output["scale"] = output["scale"].clamp(1, 1)
        if "opacity" in self.factors:
            output["opacity"] = torch.sigmoid(output["opacity"]) * self.factors[
                "opacity"
            ] + (1 - self.factors["opacity"])

        return output


class ModLinear(torch.nn.Module):
    r"""Linear layer with affine modulation (Based on StyleGAN2 mod demod).
    Equivalent to affine modulation following linear, but faster when the same modulation parameters are shared across
    multiple inputs.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        style_features (int): Number of style features.
        bias (bool): Apply additive bias before the activation function?
        mod_bias (bool): Whether to modulate bias.
        output_mode (bool): If True, modulate output instead of input.
        weight_gain (float): Initialization gain
    """

    def __init__(
        self,
        in_features,
        out_features,
        style_features,
        bias=True,
        mod_bias=True,
        output_mode=False,
        weight_gain=1,
        bias_init=0,
    ):
        super(ModLinear, self).__init__()
        weight_gain = weight_gain / np.sqrt(in_features)
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) * weight_gain
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_alpha = torch.nn.Parameter(
            torch.randn([in_features, style_features]) / np.sqrt(style_features)
        )
        self.bias_alpha = torch.nn.Parameter(
            torch.full([in_features], 1, dtype=torch.float)
        )  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        self.output_mode = output_mode
        if mod_bias:
            if output_mode:
                mod_bias_dims = out_features
            else:
                mod_bias_dims = in_features
            self.weight_beta = torch.nn.Parameter(
                torch.randn([mod_bias_dims, style_features]) / np.sqrt(style_features)
            )
            self.bias_beta = torch.nn.Parameter(
                torch.full([mod_bias_dims], 0, dtype=torch.float)
            )

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, ...   , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], -1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        w = self.weight.to(x.dtype)  # [O I]
        w = w.unsqueeze(0) * alpha
        # w = alpha @ w  # [B, ..., O]

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            if not self.output_mode:
                x = x + beta

        b = self.bias
        if b is not None:
            b = b.to(x.dtype)[None, None, :]
        if self.mod_bias and self.output_mode:
            if b is None:
                b = beta
            else:
                b = b + beta

        # [B ? I] @ [B I O] = [B ? O]
        if b is not None:
            x = torch.baddbmm(b, x, w.transpose(1, 2))
            # x = x * w + b
        else:
            x = x.bmm(w.transpose(1, 2))
            # x = x * w

        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x
