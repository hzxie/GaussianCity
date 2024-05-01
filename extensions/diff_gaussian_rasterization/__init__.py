# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Inria <george.drettakis@inria.fr>
# @Date:   2024-01-31 19:07:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-05-01 14:14:49
# @Email:  root@haozhexie.com

import math
import numpy as np
import scipy.spatial.transform
import torch
import typing

import diff_gaussian_rasterization_ext as dgr_ext


class RasterizeGaussiansFunction(torch.autograd.Function):
    @staticmethod
    def _cpu_deep_copy_tuple(input_tuple):
        copied_tensors = [
            item.cpu().clone() if isinstance(item, torch.Tensor) else item
            for item in input_tuple
        ]
        return tuple(copied_tensors)

    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.view_matrix,
            raster_settings.proj_matrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.img_h,
            raster_settings.img_w,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = RasterizeGaussiansFunction._cpu_deep_copy_tuple(
                input_tuple=args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geom_buffer,
                    binning_buffer,
                    img_buffer,
                ) = dgr_ext.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geom_buffer,
                binning_buffer,
                img_buffer,
            ) = dgr_ext.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geom_buffer,
            binning_buffer,
            img_buffer,
        )
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geom_buffer,
            binning_buffer,
            img_buffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.view_matrix,
            raster_settings.proj_matrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geom_buffer,
            num_rendered,
            binning_buffer,
            img_buffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = RasterizeGaussiansFunction._cpu_deep_copy_tuple(
                input_tuple=args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = dgr_ext.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
            ) = dgr_ext.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(typing.NamedTuple):
    img_h: int
    img_w: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    view_matrix: torch.Tensor
    proj_matrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(torch.nn.Module):
    def __init__(self, raster_settings):
        super(GaussianRasterizer, self).__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return RasterizeGaussiansFunction.apply(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )


class GaussianRasterizerWrapper(torch.nn.Module):
    # Carving flowers on a mountain of dung code.
    #
    # This class is a wrapper for the GaussianRasterizer class.
    # It is used to port for the GaussianCity project.
    def __init__(
        self,
        K,
        sensor_size,
        flip_lr=True,
        flip_ud=False,
        z_near=0.01,
        z_far=50000.0,
        device=torch.device("cuda"),
    ):
        super(GaussianRasterizerWrapper, self).__init__()
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.z_near = z_near
        self.z_far = z_far
        self.device = device
        # Shared camera parameters
        self.K = K
        self.sensor_size = sensor_size
        self.fov_x, self.fov_y = self._intrinsic_to_fov()
        self.P = self._get_projection_matrix()

    def get_gaussian_rasterizer(self, cam_position, cam_quaternion):
        # cam_position in (tx, ty, tz)
        # cam_quaternion in (qx, qy, qz, qw)
        return GaussianRasterizer(
            raster_settings=self._get_gaussian_rasterization_settings(
                cam_position, cam_quaternion
            )
        )

    def forward(
        self, points, cam_position=None, cam_quaternion=None, gaussian_rasterizer=None
    ):
        # points: [N, M], M -> 0:3 xyz, 3:4 opacity, 4:7 scale, 7:11 rotation, 11:14 rgbs
        _, M = points.shape
        assert M == 14, "The input tensor should have 14 channels."

        if gaussian_rasterizer is None:
            gaussian_rasterizer = self.get_gaussian_rasterizer(
                cam_position, cam_quaternion
            )

        return self._get_gaussian_rasterization(points, gaussian_rasterizer)

    def _intrinsic_to_fov(self):
        # graphdeco-inria/gaussian-splatting/utils/graphics_utils.py#L76
        fx, fy = self.K[0, 0], self.K[1, 1]
        fov_x = 2 * np.arctan2(self.sensor_size[0], (2 * fx))
        fov_y = 2 * np.arctan2(self.sensor_size[1], (2 * fy))
        return fov_x, fov_y

    def _get_projection_matrix(self):
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2.0 * fx / self.sensor_size[0]
        P[1, 1] = 2.0 * fy / self.sensor_size[1]
        P[0, 2] = (2.0 * cx / self.sensor_size[0]) - 1.0
        P[1, 2] = (2.0 * cy / self.sensor_size[1]) - 1.0
        P[2, 2] = -(self.z_far + self.z_near) / (self.z_far - self.z_near)
        P[3, 2] = -1.0
        P[2, 3] = -2.0 * self.z_far * self.z_near / (self.z_far - self.z_near)
        return torch.from_numpy(P).to(self.device)

    def _get_w2c_matrix(self, cam_position, cam_quaternion):
        if type(cam_position) is torch.Tensor:
            cam_position = cam_position.cpu().numpy()
        if type(cam_quaternion) is torch.Tensor:
            cam_quaternion = cam_quaternion.cpu().numpy()

        R = scipy.spatial.transform.Rotation.from_quat(cam_quaternion).as_matrix()
        # look_at = cam_position + R[:3, 0]
        R = R[:, [1, 2, 0]]  # [F|R|U] -> [R|U|F]
        # graphdeco-inria/gaussian-splatting/blob/main/scene/cameras.py#L31
        # The w2c matrix
        Rt = np.zeros((4, 4), dtype=np.float32)
        Rt[:3, :3] = R.transpose()
        Rt[:3, [3]] = -R.transpose() @ cam_position[:, None]
        Rt[3, 3] = 1.0
        # The c2w matrix
        # Rt[:3, :3] = R
        # Rt[:3, 3] = cam_position
        # Rt[3, 3] = 1.0
        return torch.from_numpy(Rt).to(self.device)

    def _world_to_pixel(self, world_coords, w2c):
        # NOTE: The function is used to debug whether the w2c matrix is correct.
        # Convert world coordinates to camera coordinates using the inverse of w2c
        camera_coords = np.dot(w2c[:3, :3], world_coords) + w2c[:3, 3]
        # camera_coords = np.dot(np.linalg.inv(c2w[:3, :3]), (world_coords- w2c[:3, 3]))
        # Apply the camera intrinsic matrix K to obtain normalized image coordinates
        homogeneous_coords = np.dot(self.K, camera_coords)
        # Normalize homogeneous coordinates
        normalized_coords = homogeneous_coords / homogeneous_coords[2]
        # Convert normalized coordinates to pixel coordinates
        return normalized_coords[:2].astype(int)

    def _get_gaussian_rasterization_settings(self, cam_position, cam_quaternion):
        BG_COLOR = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )
        w2c = self._get_w2c_matrix(cam_position, cam_quaternion).transpose(0, 1)
        prj_mtx = self.P.transpose(0, 1)

        return GaussianRasterizationSettings(
            img_h=self.sensor_size[1],
            img_w=self.sensor_size[0],
            tanfovx=math.tan(self.fov_x * 0.5),
            tanfovy=math.tan(self.fov_y * 0.5),
            bg=BG_COLOR,
            scale_modifier=1.0,
            view_matrix=w2c,
            proj_matrix=w2c @ prj_mtx,
            sh_degree=0,
            campos=w2c.inverse()[3, :3],
            prefiltered=False,
            debug=False,
        )

    def _get_gaussian_rasterization(self, points, rasterizer):
        xyz = points[:, 0:3]
        opacity = points[:, 3:4]
        scales = points[:, 4:7]
        quaternion = points[:, 7:11]
        rgbs = points[:, 11:]

        rendered_image, _ = rasterizer(
            means3D=xyz,
            means2D=torch.zeros_like(xyz, dtype=torch.float32, device=self.device),
            shs=None,
            colors_precomp=rgbs,
            opacities=opacity,
            scales=scales,
            rotations=quaternion,
            cov3D_precomp=None,
        )
        if self.flip_lr:
            rendered_image = torch.flip(rendered_image, dims=[2])
        if self.flip_ud:
            rendered_image = torch.flip(rendered_image, dims=[1])

        return rendered_image
