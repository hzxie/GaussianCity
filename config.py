# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-25 22:06:59
# @Email:  root@haozhexie.com

from easydict import EasyDict

# fmt: off
__C                                             = EasyDict()
cfg                                             = __C

#
# Dataset Config
#
cfg.DATASETS                                     = EasyDict()
cfg.DATASETS.CITY_SAMPLE                         = EasyDict()
cfg.DATASETS.CITY_SAMPLE.DIR                     = "./data"
cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY              = ["K", "Rt", "centers"]
cfg.DATASETS.CITY_SAMPLE.N_REPEAT                = 1
cfg.DATASETS.CITY_SAMPLE.N_CLASSES               = 9
cfg.DATASETS.CITY_SAMPLE.N_MIN_PIXELS_CROP       = 64
cfg.DATASETS.CITY_SAMPLE.N_CITIES                = 1
cfg.DATASETS.CITY_SAMPLE.N_VIEWS                 = 3000
cfg.DATASETS.CITY_SAMPLE.CITY_STYLES             = ["Day", "Night"]

#
# Constants
#
cfg.CONST                                        = EasyDict()
cfg.CONST.EXP_NAME                               = ""
cfg.CONST.N_WORKERS                              = 0
cfg.CONST.NETWORK                                = None

#
# Directories
#
cfg.DIR                                          = EasyDict()
cfg.DIR.OUTPUT                                   = "./output"

#
# Memcached
#
cfg.MEMCACHED                                    = EasyDict()
cfg.MEMCACHED.ENABLED                            = False
cfg.MEMCACHED.LIBRARY_PATH                       = "/mnt/lustre/share/pymc/py3"
cfg.MEMCACHED.SERVER_CONFIG                      = "/mnt/lustre/share/memcached_client/server_list.conf"
cfg.MEMCACHED.CLIENT_CONFIG                      = "/mnt/lustre/share/memcached_client/client.conf"

#
# WandB
#
cfg.WANDB                                        = EasyDict()
cfg.WANDB.ENABLED                                = False
cfg.WANDB.PROJECT                                = "City-Gen-HD"
cfg.WANDB.ENTITY                                 = "haozhexie"
cfg.WANDB.MODE                                   = "online"
cfg.WANDB.RUN_ID                                 = None
cfg.WANDB.SYNC_TENSORBOARD                       = False

#
# Network
#
cfg.NETWORK                                      = EasyDict()
# Gaussian
cfg.NETWORK.GAUSSIAN                             = EasyDict()
cfg.NETWORK.GAUSSIAN.USE_RGB_ONLY                = True
cfg.NETWORK.GAUSSIAN.FEATURE_DIM                 = 128
cfg.NETWORK.GAUSSIAN.INIT_OPACITY                = 0.1
cfg.NETWORK.GAUSSIAN.INIT_SCALING                = -5.0
cfg.NETWORK.GAUSSIAN.CLIP_SCALING                = 0.2
cfg.NETWORK.GAUSSIAN.CLIP_XYZ_OFFSET             = 1.0 / 32

#
# Train
#
cfg.TRAIN                                        = EasyDict()
cfg.TRAIN.GAUSSIAN                               = EasyDict()
cfg.TRAIN.GAUSSIAN.DATASET                       = "CITY_SAMPLE"
cfg.TRAIN.GAUSSIAN.BATCH_SIZE                    = 1
cfg.TRAIN.GAUSSIAN.CROP_SIZE                     = (224, 224)

#
# Test
#
cfg.TEST                                         = EasyDict()
cfg.TEST.GAUSSIAN                                = EasyDict()
cfg.TEST.GAUSSIAN.CROP_SIZE                      = (480, 270)
# fmt: on
