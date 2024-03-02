# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-02 15:06:24
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
## The following parameters should be the same as scripts/dataset_generator.py
cfg.DATASETS.CITY_SAMPLE.BLDG_RANGE              = [100, 5000]
cfg.DATASETS.CITY_SAMPLE.BLDG_FACADE_CLSID       = 7
cfg.DATASETS.CITY_SAMPLE.BLDG_ROOF_CLSID         = 8
cfg.DATASETS.CITY_SAMPLE.CAR_RANGE               = [5000, 16384]
cfg.DATASETS.CITY_SAMPLE.CAR_CLSID               = 3
cfg.DATASETS.CITY_SAMPLE.Z_SCALE_SPECIAL_CLASSES = {"ROAD": 1, "WATER": 4, "ZONE": 6}
cfg.DATASETS.CITY_SAMPLE.MAP_SIZE                = 24576
cfg.DATASETS.CITY_SAMPLE.SCALE                   = 20

#
# Constants
#
cfg.CONST                                        = EasyDict()
cfg.CONST.EXP_NAME                               = ""
cfg.CONST.N_WORKERS                              = 8
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
cfg.NETWORK.GAUSSIAN.Z_DIM                       = 256
cfg.NETWORK.GAUSSIAN.FEATURE_DIM                 = 512
cfg.NETWORK.GAUSSIAN.N_ATTENTION_HEADS           = 8
cfg.NETWORK.GAUSSIAN.N_TRANSFORMER_LAYERS        = 128

#
# Train
#
cfg.TRAIN                                        = EasyDict()
cfg.TRAIN.GAUSSIAN                               = EasyDict()
cfg.TRAIN.GAUSSIAN.DATASET                       = "CITY_SAMPLE"
cfg.TRAIN.GAUSSIAN.N_MIN_PIXELS                  = 64
cfg.TRAIN.GAUSSIAN.N_MAX_POINTS                  = 8192
cfg.TRAIN.GAUSSIAN.BATCH_SIZE                    = 1
cfg.TRAIN.GAUSSIAN.LR_GENERATOR                  = 1e-4
cfg.TRAIN.GAUSSIAN.LR_DISCRIMINATOR              = 1e-5
cfg.TRAIN.GAUSSIAN.DISCRIMINATOR_N_WARMUP_ITERS  = 100000
cfg.TRAIN.GAUSSIAN.EPS                           = 1e-7
cfg.TRAIN.GAUSSIAN.WEIGHT_DECAY                  = 0
cfg.TRAIN.GAUSSIAN.BETAS                         = (0., 0.999)
cfg.TRAIN.GAUSSIAN.CROP_SIZE                     = (336, 336)
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_MODEL         = "vgg19"
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_LAYERS        = ["relu_3_1", "relu_4_1", "relu_5_1"]
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_WEIGHTS       = [0.125, 0.25, 1.0]
cfg.TRAIN.GAUSSIAN.N_EPOCHS                      = 500
cfg.TRAIN.GAUSSIAN.L1_LOSS_FACTOR                = 10
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_FACTOR        = 10

#
# Test
#
cfg.TEST                                         = EasyDict()
cfg.TEST.GAUSSIAN                                = EasyDict()
cfg.TEST.GAUSSIAN.CROP_SIZE                      = (960, 540)
# fmt: on
