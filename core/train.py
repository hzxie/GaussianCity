# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2024-02-28 15:57:40
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-04 10:39:30
# @Email:  root@haozhexie.com

import logging
import os
import time
import torch

import extensions.diff_gaussian_rasterization as dgr
import losses.gan
import losses.perceptual
import models.gaussian
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
    val_dataset = utils.datasets.get_dataset(cfg, cfg.TRAIN.GAUSSIAN.DATASET, "val")
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
    gaussian_g = models.gaussian.GaussianGenerator(cfg, train_dataset.get_n_classes())
    gaussian_d = models.gaussian.GaussianDiscriminator(cfg, train_dataset.get_n_classes())
    if torch.cuda.is_available():
        logging.info("Start running the DDP on rank %d." % local_rank)
        gaussian_g = torch.nn.parallel.DistributedDataParallel(
            gaussian_g.to(local_rank),
            device_ids=[local_rank],
        )
        gaussian_d = torch.nn.parallel.DistributedDataParallel(
            gaussian_d.to(local_rank),
            device_ids=[local_rank],
        )
    else:
        gaussian_g.device = torch.device("cpu")
        gaussian_d.device = torch.device("cpu")

    # Set up optimizers
    optimizer_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gaussian_g.parameters()),
        lr=cfg.TRAIN.GAUSSIAN.LR_GENERATOR,
        eps=cfg.TRAIN.GAUSSIAN.EPS,
        weight_decay=cfg.TRAIN.GAUSSIAN.WEIGHT_DECAY,
        betas=cfg.TRAIN.GAUSSIAN.BETAS,
    )
    optimizer_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gaussian_d.parameters()),
        lr=cfg.TRAIN.GAUSSIAN.LR_DISCRIMINATOR,
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
        gaussian_g.load_state_dict(checkpoint["gaussian_g"])
        gaussian_d.load_state_dict(checkpoint["gaussian_d"])
        init_epoch = checkpoint["epoch_index"]
        logging.info("Recover completed. Current epoch = #%d" % (init_epoch,))

    # Set up folders for logs, snapshot and checkpoints
    if utils.distributed.is_master():
        output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
        cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
        cfg.DIR.LOGS = output_dir % "logs"
        os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)
        # Summary writer
        tb_writer = utils.summary_writer.SummaryWriter(cfg)

    # Set up the GaussianRasterizer
    gr = dgr.GaussianRasterizerWrapper(
        train_dataset.get_K(),
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
        gaussian_d.train()
        batch_end_time = time.time()
        from tqdm import tqdm

        for batch_idx, data in enumerate(tqdm(train_data_loader)):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time.time() - batch_end_time)
            # Warm up the discriminator
            if n_itr <= cfg.TRAIN.GAUSSIAN.DISCRIMINATOR_N_WARMUP_ITERS:
                lr = (
                    cfg.TRAIN.GAUSSIAN.LR_DISCRIMINATOR
                    * n_itr
                    / cfg.TRAIN.GAUSSIAN.DISCRIMINATOR_N_WARMUP_ITERS
                )
                for pg in optimizer_d.param_groups:
                    pg["lr"] = lr

            # Move data to GPU
            pts = utils.helpers.var_or_cuda(data["pts"], gaussian_g.device)
            rgb = utils.helpers.var_or_cuda(data["rgb"], gaussian_g.device)
            seg = utils.helpers.var_or_cuda(data["seg"], gaussian_g.device)
            msk = utils.helpers.var_or_cuda(data["msk"], gaussian_g.device)

            # Split pts into attributes
            abs_xyz = pts[:, :, :3]
            rel_xyz = pts[:, :, 5:]
            instances = pts[:, :, [4]]
            classes = train_dataset.instances_to_classes(instances)
            scales = pts[:, :, [3]]
            scales = utils.helpers.get_point_scales(
                scales, classes, train_dataset.get_special_z_scale_classes()
            )
            onehots = utils.helpers.get_one_hot(
                classes, train_dataset.get_n_classes()
            )
            z = utils.helpers.get_z(instances, cfg.NETWORK.GAUSSIAN.Z_DIM)
            # Make the number of points in the batch consistent
            n_pts = pts.size(1)
            n_max_pts = torch.max(data["npt"])
            pts = torch.cat([rel_xyz, onehots, z], dim=2)
            pts = utils.helpers.get_pad_tensor(pts, n_max_pts)

            # Discriminator Update Step

            # Generator Update Step
            try:
                pt_rgbs = gaussian_g(pts)
                gs_pts = utils.helpers.get_gaussian_points(
                    n_pts, abs_xyz, scales, pt_rgbs
                )
                fake_imgs = utils.helpers.get_gaussian_rasterization(
                    gs_pts, gr, data["cam_pos"], data["cam_quat"], data["crp"]
                )
                _l1_loss = l1_loss(fake_imgs * msk, rgb * msk)
                _perceptual_loss = perceptual_loss(fake_imgs * msk, rgb * msk)
                loss_g = (
                    _l1_loss * cfg.TRAIN.GAUSSIAN.L1_LOSS_FACTOR
                    + _perceptual_loss * cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_FACTOR
                )
                gaussian_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
            except Exception as ex:
                logging.warning("#Points: %d, Msg: %s" % (n_max_pts, ex))
                torch.cuda.empty_cache()
                continue
            finally:
                torch.cuda.empty_cache()
