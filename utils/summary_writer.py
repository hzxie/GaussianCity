# -*- coding: utf-8 -*-
#
# @File:   summary_writer.py
# @Author: Haozhe Xie
# @Date:   2020-04-19 12:52:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-20 09:38:29
# @Email:  root@haozhexie.com

import numpy as np
import logging
import PIL
import os
import torch.utils.tensorboard

try:
    import wandb
except Exception as ex:
    logging.warning(ex)


class SummaryWriter(object):
    def __init__(self, cfg):
        os.makedirs(cfg.DIR.OUTPUT, exist_ok=True)
        if cfg.WANDB.ENABLED:
            if cfg.WANDB.get("RUN_ID"):
                logging.info("Resuming from WandB[ID=%s]" % cfg.WANDB.RUN_ID)
            else:
                cfg.WANDB.RUN_ID = wandb.util.generate_id()

            self.writer = wandb.init(
                id=cfg.WANDB.RUN_ID,
                entity=cfg.WANDB.ENTITY,
                project=cfg.WANDB.PROJECT,
                name=cfg.CONST.EXP_NAME,
                dir=cfg.DIR.OUTPUT,
                mode=cfg.WANDB.MODE,
                resume="allow",
            )
            if cfg.WANDB.LOG_CODE:
                wandb.run.log_code(
                    os.path.join(os.path.dirname(__file__), os.path.pardir)
                )
        else:
            self.writer = torch.utils.tensorboard.SummaryWriter(cfg.DIR.LOGS)

    def add_config(self, cfg):
        if isinstance(self.writer, torch.utils.tensorboard.writer.SummaryWriter):
            logging.warning("TensorBoard does not support adding config.")
        else:
            for k, v in cfg.items():
                self.writer.config[k] = v

    def add_scalars(self, scalars, step=None):
        if isinstance(self.writer, torch.utils.tensorboard.writer.SummaryWriter):
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        else:
            self.writer.log(scalars)

    def _get_tb_image(self, image):
        # Related to: utils.helpers.tensor_to_image
        if isinstance(image, PIL.Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray) and len(image.shape) == 2:
            return image
        elif isinstance(image, np.ndarray) and len(image.shape) == 3:
            return image
        else:
            raise Exception("Unknown image format")

    def _get_tb_image_format(self, image):
        # Related to: utils.helpers.tensor_to_image
        if isinstance(image, PIL.Image.Image):
            return "HWC"
        elif isinstance(image, np.ndarray) and len(image.shape) == 2:
            return "HW"
        elif isinstance(image, np.ndarray) and len(image.shape) == 3:
            return "HWC"
        else:
            raise Exception("Unknown image format")

    def add_images(self, images, step=None):
        if isinstance(self.writer, torch.utils.tensorboard.writer.SummaryWriter):
            for k, v in images.items():
                self.writer.add_image(
                    k,
                    self._get_tb_image(v),
                    step,
                    dataformats=self._get_tb_image_format(v),
                )
        else:
            self.writer.log({k: wandb.Image(v) for k, v in images.items()})

    def close(self):
        if isinstance(self.writer, torch.utils.tensorboard.writer.SummaryWriter):
            self.writer.close()
        else:
            self.writer.finish(exit_code=0)
