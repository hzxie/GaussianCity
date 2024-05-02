# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2024-02-28 15:57:40
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-05-02 19:13:37
# @Email:  root@haozhexie.com

import logging
import os
import shutil
import time
import torch
import torch.nn.functional as F

import core.test
import extensions.diff_gaussian_rasterization as dgr
import losses.gan
import losses.perceptual
import models.generator
import models.discriminator
import utils.average_meter
import utils.datasets
import utils.distributed
import utils.helpers
import utils.summary_writer


def train(cfg):
    torch.backends.cudnn.benchmark = True
    local_rank = utils.distributed.get_rank()
    # Set up data loader
    train_dataset = utils.datasets.get_dataset(cfg, cfg.TRAIN.GAUSSIAN.DATASET, "train")
    val_dataset = utils.datasets.get_dataset(cfg, cfg.TEST.GAUSSIAN.DATASET, "val")
    train_sampler = None
    val_sampler = None
    if torch.cuda.is_available():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=local_rank, shuffle=True, drop_last=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, rank=local_rank, shuffle=False
        )

    assert cfg.TRAIN.GAUSSIAN.BATCH_SIZE == 1, "Batch size must be 1."
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.GAUSSIAN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=False,
        sampler=train_sampler,
        persistent_workers=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=False,
        sampler=val_sampler,
        persistent_workers=True,
    )

    # Set up networks
    gaussian_g = models.generator.Generator(cfg, train_dataset.get_n_classes())
    if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
        gaussian_d = models.discriminator.Discriminator(
            cfg, train_dataset.get_n_classes()
        )
    if torch.cuda.is_available():
        logging.info("Start running the DDP on rank %d." % local_rank)
        gaussian_g = torch.nn.parallel.DistributedDataParallel(
            gaussian_g.to(local_rank),
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
        if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
            gaussian_d = torch.nn.parallel.DistributedDataParallel(
                gaussian_d.to(local_rank),
                device_ids=[local_rank],
            )
    else:
        gaussian_g.device = torch.device("cpu")
        if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
            gaussian_d.device = torch.device("cpu")

    # Set up optimizers
    optimizer_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gaussian_g.parameters()),
        lr=cfg.TRAIN.GAUSSIAN.GENERATOR.LR,
        eps=cfg.TRAIN.GAUSSIAN.EPS,
        weight_decay=cfg.TRAIN.GAUSSIAN.WEIGHT_DECAY,
        betas=cfg.TRAIN.GAUSSIAN.BETAS,
    )
    if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
        optimizer_d = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gaussian_d.parameters()),
            lr=cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.LR,
            eps=cfg.TRAIN.GAUSSIAN.EPS,
            weight_decay=cfg.TRAIN.GAUSSIAN.WEIGHT_DECAY,
            betas=cfg.TRAIN.GAUSSIAN.BETAS,
        )

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()
    gan_loss = losses.gan.GANLoss()
    perceptual_loss = losses.perceptual.PerceptualLoss(
        cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_MODEL,
        cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_LAYERS,
        cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_WEIGHTS,
        device=gaussian_g.device,
    )

    # Load the pretrained model if exists
    init_epoch = 0
    if "CKPT" in cfg.CONST:
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT, map_location=gaussian_g.device)
        # init_epoch = checkpoint["epoch_index"]
        gaussian_g.load_state_dict(checkpoint["gaussian_g"])
        if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
            gaussian_d.load_state_dict(checkpoint["gaussian_d"])
        logging.info("Recover completed. Current epoch = #%d" % (init_epoch,))

    # Set up folders for logs, snapshot and checkpoints
    if utils.distributed.is_master():
        output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
        cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
        cfg.DIR.LOGS = output_dir % "logs"
        os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)
        # Summary writer
        tb_writer = utils.summary_writer.SummaryWriter(cfg)
        # Log current config
        tb_writer.add_config(cfg.NETWORK.GAUSSIAN)
        tb_writer.add_config(cfg.TRAIN.GAUSSIAN)

    # Set up the GaussianRasterizer
    gr = dgr.GaussianRasterizerWrapper(
        K=train_dataset.get_K(),
        sensor_size=train_dataset.get_sensor_size(),
        flip_ud=train_dataset.is_flip_ud(),
        device=gaussian_g.device,
    )

    # Training/Testing the network
    n_batches = len(train_data_loader)
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.GAUSSIAN.N_EPOCHS + 1):
        epoch_start_time = time.time()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        train_losses = utils.average_meter.AverageMeter(
            [
                "L1Loss",
                "PerceptualLoss",
                "GANLoss",
                "GANLossFake",
                "GANLossReal",
                "GenLoss",
                "DisLoss",
            ]
        )
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        # Switch models to train mode
        gaussian_g.train()
        if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
            gaussian_d.train()

        batch_end_time = time.time()
        for batch_idx, data in enumerate(train_data_loader):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time.time() - batch_end_time)
            # Warm up the discriminator
            if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
                if n_itr <= cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.N_WARMUP_ITERS:
                    lr = (
                        cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.LR
                        * n_itr
                        / cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.N_WARMUP_ITERS
                    )
                    for pg in optimizer_d.param_groups:
                        pg["lr"] = lr

            torch.cuda.empty_cache()
            # Move data to GPU
            pts = utils.helpers.var_or_cuda(data["pts"], gaussian_g.device)
            rgb = utils.helpers.var_or_cuda(data["rgb"], gaussian_g.device)
            seg = utils.helpers.var_or_cuda(data["seg"], gaussian_g.device)
            proj_hf = utils.helpers.var_or_cuda(data["proj/hf"], gaussian_g.device)
            proj_seg = utils.helpers.var_or_cuda(data["proj/seg"], gaussian_g.device)
            proj_tlp = (
                utils.helpers.var_or_cuda(data["proj/tlp"], gaussian_g.device)
                if "proj/tlp" in data
                else None
            )
            proj_aff_mat = (
                utils.helpers.var_or_cuda(data["proj/affmat"], gaussian_g.device)
                if "proj/tlp" in data
                else None
            )
            msk = utils.helpers.var_or_cuda(data["msk"], gaussian_g.device)
            gan_loss_weights = F.interpolate(msk, scale_factor=0.25)

            # Split pts into attributes
            abs_xyz = pts[:, :, :3]
            rel_xyz = pts[:, :, 5:8]
            bch_idx = pts[:, :, 8].long()
            instances = pts[:, :, [4]]
            classes = train_dataset.instances_to_classes(instances)
            scales = pts[:, :, [3]] * cfg.NETWORK.GAUSSIAN.SCALE_FACTOR
            scales = utils.helpers.get_point_scales(
                scales,
                classes,
                train_dataset.get_special_z_scale_classes(),
            )
            onehots = utils.helpers.get_one_hot(classes, train_dataset.get_n_classes())
            z = utils.helpers.get_z(instances, cfg.NETWORK.GAUSSIAN.Z_DIM)
            # Points positions at projection maps
            proj_size = train_dataset.get_proj_size()
            proj_uv = utils.helpers.get_projection_uv(
                abs_xyz, proj_tlp, proj_aff_mat, proj_size
            )

            # Discriminator Update Step
            if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
                utils.helpers.requires_grad(gaussian_g, False)
                utils.helpers.requires_grad(gaussian_d, True)

                with torch.no_grad():
                    pt_attrs = gaussian_g(
                        proj_uv, rel_xyz, bch_idx, onehots, z, proj_hf, proj_seg
                    )
                    gs_pts = utils.helpers.get_gaussian_points(
                        abs_xyz, scales.clone(), pt_attrs
                    )
                    fake_imgs = utils.helpers.get_gaussian_rasterization(
                        gs_pts,
                        gr,
                        data["cam_pos"],
                        data["cam_quat"],
                        data["crp"] if "crp" in data else None,
                    ).detach()

                fake_labels = gaussian_d(fake_imgs, seg, msk)
                real_labels = gaussian_d(rgb, seg, msk)
                fake_loss = gan_loss(
                    fake_labels, False, gan_loss_weights, dis_update=True
                )
                real_loss = gan_loss(
                    real_labels, True, gan_loss_weights, dis_update=True
                )
                loss_d = fake_loss + real_loss
                gaussian_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
            else:
                fake_loss = torch.tensor(0)
                real_loss = torch.tensor(0)
                loss_d = torch.tensor(0)

            # Generator Update Step
            if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
                utils.helpers.requires_grad(gaussian_d, False)
                utils.helpers.requires_grad(gaussian_g, True)

            pt_attrs = gaussian_g(
                proj_uv, rel_xyz, bch_idx, onehots, z, proj_hf, proj_seg
            )
            gs_pts = utils.helpers.get_gaussian_points(
                abs_xyz, scales.clone(), pt_attrs
            )
            fake_imgs = utils.helpers.get_gaussian_rasterization(
                gs_pts, gr, data["cam_pos"], data["cam_quat"], data["crp"]
            )
            if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
                fake_labels = gaussian_d(fake_imgs, seg, msk)
                _gan_loss = gan_loss(
                    fake_labels, True, gan_loss_weights, dis_update=False
                )
            else:
                _gan_loss = torch.tensor(0)

            _l1_loss = l1_loss(fake_imgs * msk, rgb * msk)
            _perceptual_loss = perceptual_loss(fake_imgs * msk, rgb * msk)
            loss_g = (
                _l1_loss * cfg.TRAIN.GAUSSIAN.L1_LOSS_FACTOR
                + _perceptual_loss * cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_FACTOR
                + _gan_loss * cfg.TRAIN.GAUSSIAN.GAN_LOSS_FACTOR
            )

            gaussian_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            train_losses.update(
                [
                    _l1_loss.item(),
                    _perceptual_loss.item(),
                    _gan_loss.item(),
                    fake_loss.item(),
                    real_loss.item(),
                    loss_g.item(),
                    loss_d.item(),
                ]
            )

            batch_time.update(time.time() - batch_end_time)
            batch_end_time = time.time()
            if utils.distributed.is_master():
                tb_writer.add_scalars(
                    {
                        "Loss/Batch/L1": train_losses.val(0),
                        "Loss/Batch/Perceptual": train_losses.val(1),
                        "Loss/Batch/GAN": train_losses.val(2),
                        "Loss/Batch/GANFake": train_losses.val(3),
                        "Loss/Batch/GANReal": train_losses.val(4),
                        "Loss/Batch/GenTotal": train_losses.val(5),
                        "Loss/Batch/DisTotal": train_losses.val(6),
                    },
                    n_itr,
                )
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s"
                    % (
                        epoch_idx,
                        cfg.TRAIN.GAUSSIAN.N_EPOCHS,
                        batch_idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ["%.4f" % l for l in train_losses.val()],
                    )
                )

        epoch_end_time = time.time()
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "Loss/Epoch/L1/Train": train_losses.avg(0),
                    "Loss/Epoch/Perceptual/Train": train_losses.avg(1),
                    "Loss/Epoch/GAN/Train": train_losses.avg(2),
                    "Loss/Epoch/GANFake/Train": train_losses.avg(3),
                    "Loss/Epoch/GANReal/Train": train_losses.avg(4),
                    "Loss/Epoch/GenTotal/Train": train_losses.avg(5),
                    "Loss/Epoch/DisTotal/Train": train_losses.avg(6),
                },
                epoch_idx,
            )
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s"
                % (
                    epoch_idx,
                    cfg.TRAIN.GAUSSIAN.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    ["%.4f" % l for l in train_losses.avg()],
                )
            )

        # Evaluate the current model
        test_losses, key_frames = core.test(
            cfg,
            val_data_loader,
            gaussian_g,
        )
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "Loss/Epoch/L1/Test": test_losses.avg(0),
                },
                epoch_idx,
            )
            tb_writer.add_images(key_frames, epoch_idx)
            # Save ckeckpoints
            logging.info("Saved checkpoint to ckpt-last.pth ...")
            ckpt = {
                "cfg": cfg,
                "epoch_index": epoch_idx,
                "gaussian_g": gaussian_g.state_dict(),
            }
            if cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED:
                ckpt["gaussian_d"] = gaussian_d.state_dict()

            torch.save(
                ckpt,
                os.path.join(cfg.DIR.CHECKPOINTS, "ckpt-last.pth"),
            )
            if epoch_idx % cfg.TRAIN.GAUSSIAN.CKPT_SAVE_FREQ == 0:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "ckpt-last.pth"),
                    os.path.join(
                        cfg.DIR.CHECKPOINTS, "ckpt-epoch-%03d.pth" % epoch_idx
                    ),
                )

    if utils.distributed.is_master():
        tb_writer.close()
