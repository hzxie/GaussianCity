# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-28 14:50:44
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
cfg.DATASETS.CITY_SAMPLE.DIR                     = "./data/city-sample"
cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY              = ["Rt", "centers"]
cfg.DATASETS.CITY_SAMPLE.CAM_K                   = [2828.2831640142235, 0, 960, 0, 2828.2831640142235, 540, 0, 0, 1]
cfg.DATASETS.CITY_SAMPLE.N_REPEAT                = 1
cfg.DATASETS.CITY_SAMPLE.N_CLASSES               = 9
cfg.DATASETS.CITY_SAMPLE.N_CITIES                = 1
cfg.DATASETS.CITY_SAMPLE.N_VIEWS                 = 3000
cfg.DATASETS.CITY_SAMPLE.CITY_STYLES             = ["Day", "Night"]
## The following parameters is for training efficiency
cfg.DATASETS.CITY_SAMPLE.N_MIN_PIXELS_CROP       = 64
cfg.DATASETS.CITY_SAMPLE.N_MAX_POINTS_CROP       = 16384
## The following parameters should be the same as scripts/dataset_generator.py
cfg.DATASETS.CITY_SAMPLE.MAP_SIZE                = 24576
cfg.DATASETS.CITY_SAMPLE.SCALE                   = 20

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
cfg.WANDB.PROJECT                                = "GaussianCity"
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
cfg.TRAIN.GAUSSIAN.CROP_SIZE                     = (336, 336)

#
# Test
#
cfg.TEST                                         = EasyDict()
cfg.TEST.GAUSSIAN                                = EasyDict()
cfg.TEST.GAUSSIAN.CROP_SIZE                      = (960, 540)
# fmt: on
