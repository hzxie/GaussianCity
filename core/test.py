# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2024-02-28 15:58:23
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-09-18 14:41:29
# @Email:  root@haozhexie.com

import logging
import torch

import extensions.diff_gaussian_rasterization as dgr
import models.generator
import utils.average_meter
import utils.datasets
import utils.helpers

from tqdm import tqdm


def test(cfg, test_data_loader=None, gaussian_g=None):
    torch.backends.cudnn.benchmark = True
    if test_data_loader is None:
        test_data_loader = torch.utils.data.DataLoader(
            dataset=utils.datasets.get_dataset(cfg, cfg.CONST.DATASET, "test"),
            batch_size=1,
            num_workers=cfg.CONST.N_WORKERS,
            collate_fn=utils.datasets.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    if gaussian_g is None:
        gaussian_g = models.generator.Generator(
            cfg.NETWORK.GAUSSIAN,
            test_data_loader.dataset.get_n_classes(),
            test_data_loader.dataset.get_proj_size(),
        )
        if torch.cuda.is_available():
            gaussian_g = torch.nn.DataParallel(gaussian_g).cuda()
            gaussian_g.device = gaussian_g.output_device

        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        gaussian_g.load_state_dict(checkpoint["gaussian_g"])

    # Switch models to evaluation mode
    gaussian_g.eval()

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()

    # Set up the GaussianRasterizer
    gr = dgr.GaussianRasterizerWrapper(
        K=test_data_loader.dataset.get_K(),
        sensor_size=test_data_loader.dataset.get_sensor_size(),
        flip_ud=test_data_loader.dataset.is_flip_ud(),
        device=gaussian_g.device,
    )

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = utils.average_meter.AverageMeter(["L1Loss"])
    key_frames = {}
    for idx, data in enumerate(tqdm(test_data_loader)):
        with torch.no_grad():
            pts = utils.helpers.var_or_cuda(data["pts"], gaussian_g.device)
            rgb = utils.helpers.var_or_cuda(data["rgb"], gaussian_g.device)
            proj_hf = utils.helpers.var_or_cuda(data["proj/hf"], gaussian_g.device)
            proj_seg = utils.helpers.var_or_cuda(data["proj/seg"], gaussian_g.device)
            proj_tlp = (
                utils.helpers.var_or_cuda(data["proj/tlp"], gaussian_g.device)
                if "proj/tlp" in data
                else None
            )

            # Split pts into attributes
            abs_xyz = pts[:, :, :3]
            rel_xyz = pts[:, :, 5:8]
            bch_idx = pts[:, :, 8].long()
            instances = pts[:, :, [4]]
            classes = test_data_loader.dataset.instances_to_classes(instances)
            scales = pts[:, :, [3]] * cfg.NETWORK.GAUSSIAN.SCALE_FACTOR
            scales = utils.helpers.get_point_scales(
                scales,
                classes,
                test_data_loader.dataset.get_special_z_scale_classes(),
            )
            onehots = utils.helpers.get_one_hot(
                classes, test_data_loader.dataset.get_n_classes()
            )
            z = utils.helpers.get_z(instances, cfg.NETWORK.GAUSSIAN.Z_DIM)
            # Points positions at projection maps
            proj_uv = utils.helpers.get_projection_uv(
                abs_xyz, proj_tlp, test_data_loader.dataset.get_proj_size()
            )

            pt_rgbs = gaussian_g(
                proj_uv, rel_xyz, bch_idx, onehots, z, proj_hf, proj_seg
            )
            gs_pts = utils.helpers.get_gaussian_points(abs_xyz, scales, pt_rgbs)
            fake_imgs = utils.helpers.get_gaussian_rasterization(
                gs_pts,
                gr,
                data["cam_pos"],
                data["cam_quat"],
                data["crp"] if "crp" in data else None,
            )
            loss = l1_loss(fake_imgs, rgb)
            test_losses.update([loss.item()])

            if idx % cfg.TEST.GAUSSIAN.TEST_FREQ != 0:
                continue

            if utils.distributed.is_master():
                key_frames["Image/%04d" % idx] = utils.helpers.tensor_to_image(
                    torch.cat([fake_imgs, rgb], dim=3), "RGB"
                )
                logging.info(
                    "Test[%d/%d] Losses = %s"
                    % (idx + 1, n_samples, ["%.4f" % l for l in test_losses.val()])
                )

    return test_losses, key_frames
