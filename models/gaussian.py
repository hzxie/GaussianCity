# -*- coding: utf-8 -*-
#
# @File:   gaussian.py
# @Author: Haozhe Xie
# @Date:   2024-01-31 14:11:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-31 17:20:23
# @Email:  root@haozhexie.com
#
# References:
# - https://github.com/graphdeco-inria/gaussian-splatting

import logging
import numpy as np
import torch
import typing

import extensions.simple_knn


class BasicPointCloud(typing.NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class Activations:
    @classmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = Activations._build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = Activations._strip_symmetric(actual_covariance)
        return symm

    @classmethod
    def _build_scaling_rotation(s, r):
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        R = Activations._build_rotation(r)
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        L = R @ L
        return L

    @classmethod
    def _build_rotation(r):
        norm = torch.sqrt(
            r[:, 0] * r[:, 0]
            + r[:, 1] * r[:, 1]
            + r[:, 2] * r[:, 2]
            + r[:, 3] * r[:, 3]
        )
        q = r / norm[:, None]
        R = torch.zeros((q.size(0), 3, 3), device="cuda")
        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    @classmethod
    def _strip_symmetric(sym):
        uncertainty = torch.zeros((sym.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = sym[:, 0, 0]
        uncertainty[:, 1] = sym[:, 0, 1]
        uncertainty[:, 2] = sym[:, 0, 2]
        uncertainty[:, 3] = sym[:, 1, 1]
        uncertainty[:, 4] = sym[:, 1, 2]
        uncertainty[:, 5] = sym[:, 2, 2]
        return uncertainty

    @classmethod
    def inverse_sigmoid(x):
        return torch.log(x / (1 - x))


class SHUtility:
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    @classmethod
    def rgb_to_sh(rgb):
        return (rgb - 0.5) / SHUtility.C0

    @classmethod
    def sh_to_rgb(sh):
        return sh * SHUtility.C0 + 0.5

    @classmethod
    def eval_sh(deg, sh, dirs):
        """
        Evaluate spherical harmonics at unit directions
        using hardcoded SH polynomials.
        Works with torch/np/jnp.
        ... Can be 0 or more batch dimensions.
        Args:
            deg: int SH deg. Currently, 0-3 supported
            sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
            dirs: jnp.ndarray unit directions [..., 3]
        Returns:
            [..., C]
        """
        assert deg <= 4 and deg >= 0
        coeff = (deg + 1) ** 2
        assert sh.shape[-1] >= coeff

        result = SHUtility.C0 * sh[..., 0]
        if deg > 0:
            x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
            result = (
                result
                - SHUtility.C1 * y * sh[..., 1]
                + SHUtility.C1 * z * sh[..., 2]
                - SHUtility.C1 * x * sh[..., 3]
            )
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result = (
                    result
                    + SHUtility.C2[0] * xy * sh[..., 4]
                    + SHUtility.C2[1] * yz * sh[..., 5]
                    + SHUtility.C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                    + SHUtility.C2[3] * xz * sh[..., 7]
                    + SHUtility.C2[4] * (xx - yy) * sh[..., 8]
                )
                if deg > 2:
                    result = (
                        result
                        + SHUtility.C3[0] * y * (3 * xx - yy) * sh[..., 9]
                        + SHUtility.C3[1] * xy * z * sh[..., 10]
                        + SHUtility.C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                        + SHUtility.C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                        + SHUtility.C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                        + SHUtility.C3[5] * z * (xx - yy) * sh[..., 14]
                        + SHUtility.C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                    )
                    if deg > 3:
                        result = (
                            result
                            + SHUtility.C4[0] * xy * (xx - yy) * sh[..., 16]
                            + SHUtility.C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                            + SHUtility.C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                            + SHUtility.C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                            + SHUtility.C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                            + SHUtility.C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                            + SHUtility.C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                            + SHUtility.C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                            + SHUtility.C4[8]
                            * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                            * sh[..., 24]
                        )
        return result


class GaussianModel:
    @property
    def scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def xyz(self):
        return self._xyz

    @property
    def features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii_2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._setup_functions()

    def _setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = Activations.build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = Activations.inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def _training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = self._get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def _get_expon_lr_func(
        self, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
    ):
        def helper(step):
            if lr_init == lr_final:
                # constant lr, ignore other params
                return lr_init
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0

            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii_2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii_2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self._training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.scaling, scaling_modifier, self._rotation
        )

    def incerase_sh_degree(self, v=1):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += v

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        logging.info(
            "Number of points at initialization: %d" % fused_point_cloud.shape[0]
        )

        fused_color = SHUtility.rgb_to_sh(
            torch.tensor(np.asarray(pcd.colors)).float().cuda()
        )
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(
            extensions.simple_knn.dist(
                torch.from_numpy(np.asarray(pcd.points)).float().cuda()
            ),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = Activations.inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = torch.nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
        self.max_radii_2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def _construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))

        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def get_ply(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self._construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        return elements

    def load_ply(self, data):
        xyz = np.stack(
            (
                np.asarray(data.elements[0]["x"]),
                np.asarray(data.elements[0]["y"]),
                np.asarray(data.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(data.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(data.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(data.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(data.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in data.elements[0].properties if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(data.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name for p in data.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(data.elements[0][attr_name])

        rot_names = [
            p.name for p in data.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(data.elements[0][attr_name])

        self._xyz = torch.nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = torch.nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = torch.nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = torch.nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = torch.nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = torch.nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree

    def reset_opacity(self):
        opacities_new = Activations.inverse_sigmoid(
            torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01)
        )
        optimizable_tensors = self._replace_tensor_to_optimizer(
            opacities_new, "opacity"
        )
        self._opacity = optimizable_tensors["opacity"]

    def _replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii_2D = self.max_radii_2D[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self._cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.max_radii_2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def _cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = Activations._(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii_2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
