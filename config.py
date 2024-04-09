# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-04-09 14:58:55
# @Email:  root@haozhexie.com

from easydict import EasyDict

# fmt: off
__C                                               = EasyDict()
cfg                                               = __C

#
# Dataset Config
#
cfg.DATASETS                                      = EasyDict()
# The GoogleEarth Dataset Config
cfg.DATASETS.GOOGLE_EARTH                         = EasyDict()
cfg.DATASETS.GOOGLE_EARTH.DIR                     = "./data/google-earth"
cfg.DATASETS.GOOGLE_EARTH.PIN_MEMORY              = ["Rt", "centers"]
cfg.DATASETS.GOOGLE_EARTH.CAM_K                   = [1528.1469407006614, 0, 480, 0, 1528.1469407006614, 270, 0, 0, 1]
cfg.DATASETS.GOOGLE_EARTH.N_CLASSES               = 8
cfg.DATASETS.GOOGLE_EARTH.PROJ_SIZE               = (2048, 2048)
cfg.DATASETS.GOOGLE_EARTH.N_REPEAT                = 1            # 1
cfg.DATASETS.GOOGLE_EARTH.N_CITIES                = 400          # 400
cfg.DATASETS.GOOGLE_EARTH.N_VIEWS                 = 60           # 60
cfg.DATASETS.GOOGLE_EARTH.TRAIN_CROP_SIZE         = (448, 448)
cfg.DATASETS.GOOGLE_EARTH.TEST_CROP_SIZE          = (720, 405)
cfg.DATASETS.GOOGLE_EARTH.TRAIN_MIN_PIXELS        = 64
cfg.DATASETS.GOOGLE_EARTH.TRAIN_MAX_POINTS        = 16384
cfg.DATASETS.GOOGLE_EARTH.BLDG_SCALE_FACTOR       = 0.75
## The following parameters should be the same as scripts/dataset_generator.py
cfg.DATASETS.GOOGLE_EARTH.BLDG_RANGE              = [100, 32767]
cfg.DATASETS.GOOGLE_EARTH.BLDG_FACADE_CLSID       = 2
cfg.DATASETS.GOOGLE_EARTH.BLDG_ROOF_CLSID         = 7
cfg.DATASETS.GOOGLE_EARTH.Z_SCALE_SPECIAL_CLASSES = {"ROAD": 1, "WATER": 5, "ZONE": 6}
cfg.DATASETS.GOOGLE_EARTH.MAP_SIZE                = 2048
cfg.DATASETS.GOOGLE_EARTH.SCALE                   = 1
# The CitySample Dataset Config
cfg.DATASETS.CITY_SAMPLE                          = EasyDict()
cfg.DATASETS.CITY_SAMPLE.DIR                      = "./data/city-sample"
cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY               = ["Rt", "centers"]
cfg.DATASETS.CITY_SAMPLE.CAM_K                    = [2828.2831640142235, 0, 960, 0, 2828.2831640142235, 540, 0, 0, 1]
cfg.DATASETS.CITY_SAMPLE.N_CLASSES                = 9
cfg.DATASETS.CITY_SAMPLE.PROJ_SIZE                = (2048, 2048)
cfg.DATASETS.CITY_SAMPLE.N_REPEAT                 = 1            # 1
cfg.DATASETS.CITY_SAMPLE.N_CITIES                 = 1            # 10
cfg.DATASETS.CITY_SAMPLE.N_VIEWS                  = 3000         # 3000
cfg.DATASETS.CITY_SAMPLE.CITY_STYLES              = ["Day"]      # ["Day", "Night"]
cfg.DATASETS.CITY_SAMPLE.TRAIN_CROP_SIZE          = (448, 448)
cfg.DATASETS.CITY_SAMPLE.TRAIN_MIN_PIXELS         = 64
cfg.DATASETS.CITY_SAMPLE.TRAIN_MAX_POINTS         = 16384
cfg.DATASETS.CITY_SAMPLE.TEST_CROP_SIZE           = (960, 540)
cfg.DATASETS.CITY_SAMPLE.BLDG_SCALE_FACTOR        = 0.75
## The following parameters should be the same as scripts/dataset_generator.py
cfg.DATASETS.CITY_SAMPLE.BLDG_RANGE               = [100, 5000]
cfg.DATASETS.CITY_SAMPLE.BLDG_FACADE_CLSID        = 7
cfg.DATASETS.CITY_SAMPLE.BLDG_ROOF_CLSID          = 8
cfg.DATASETS.CITY_SAMPLE.CAR_RANGE                = [5000, 16384]
cfg.DATASETS.CITY_SAMPLE.CAR_CLSID                = 3
cfg.DATASETS.CITY_SAMPLE.Z_SCALE_SPECIAL_CLASSES  = {"ROAD": 1, "WATER": 4, "ZONE": 6}
cfg.DATASETS.CITY_SAMPLE.MAP_SIZE                 = 24576
cfg.DATASETS.CITY_SAMPLE.SCALE                    = 20

#
# Constants
#
cfg.CONST                                         = EasyDict()
cfg.CONST.EXP_NAME                                = ""
cfg.CONST.N_WORKERS                               = 8

#
# Directories
#
cfg.DIR                                           = EasyDict()
cfg.DIR.OUTPUT                                    = "./output"

#
# Memcached
#
cfg.MEMCACHED                                     = EasyDict()
cfg.MEMCACHED.ENABLED                             = False
cfg.MEMCACHED.LIBRARY_PATH                        = "/mnt/lustre/share/pymc/py3"
cfg.MEMCACHED.SERVER_CONFIG                       = "/mnt/lustre/share/memcached_client/server_list.conf"
cfg.MEMCACHED.CLIENT_CONFIG                       = "/mnt/lustre/share/memcached_client/client.conf"

#
# WandB
#
cfg.WANDB                                         = EasyDict()
cfg.WANDB.ENABLED                                 = False
cfg.WANDB.PROJECT                                 = "GaussianCity"
cfg.WANDB.ENTITY                                  = "haozhexie"
cfg.WANDB.MODE                                    = "online"
cfg.WANDB.RUN_ID                                  = None
cfg.WANDB.LOG_CODE                                = True
cfg.WANDB.SYNC_TENSORBOARD                        = False

#
# Network
#
cfg.NETWORK                                       = EasyDict()
# Gaussian
cfg.NETWORK.GAUSSIAN                              = EasyDict()
cfg.NETWORK.GAUSSIAN.SCALE_FACTOR                 = 0.75
cfg.NETWORK.GAUSSIAN.REPEAT_PTS                   = 4
cfg.NETWORK.GAUSSIAN.PROJ_ENCODER_OUT_DIM         = 64
cfg.NETWORK.GAUSSIAN.N_FREQ_BANDS                 = 10
cfg.NETWORK.GAUSSIAN.Z_DIM                        = 256
cfg.NETWORK.GAUSSIAN.MLP_HIDDEN_DIM               = 512
cfg.NETWORK.GAUSSIAN.MLP_N_SHARED_LAYERS          = 1
cfg.NETWORK.GAUSSIAN.ATTR_FACTORS                 = {"rgb": 2}
cfg.NETWORK.GAUSSIAN.ATTR_N_LAYERS                = {"rgb": 1}
cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE           = 128
cfg.NETWORK.GAUSSIAN.PTV3                         = EasyDict()
cfg.NETWORK.GAUSSIAN.PTV3.STRIDE                  = (2, 2, 2, 2)
cfg.NETWORK.GAUSSIAN.PTV3.ENC_DEPTHS              = (2, 2, 2, 6, 2)
cfg.NETWORK.GAUSSIAN.PTV3.ENC_CHANNELS            = (32, 64, 128, 256, 512)
cfg.NETWORK.GAUSSIAN.PTV3.ENC_N_HEAD              = (2, 4, 8, 16, 32)
cfg.NETWORK.GAUSSIAN.PTV3.ENC_PATCH_SIZE          = (1024, 1024, 1024, 1024, 1024)
cfg.NETWORK.GAUSSIAN.PTV3.DEC_DEPTHS              = (2, 2, 2, 2)
cfg.NETWORK.GAUSSIAN.PTV3.DEC_CHANNELS            = (64, 64, 128, 256)
cfg.NETWORK.GAUSSIAN.PTV3.DEC_N_HEAD              = (4, 4, 8, 16)
cfg.NETWORK.GAUSSIAN.PTV3.DEC_PATCH_SIZE          = (1024, 1024, 1024, 1024)
cfg.NETWORK.GAUSSIAN.PTV3.ENABLE_FLASH_ATTN       = False

#
# Train
#
cfg.TRAIN                                         = EasyDict()
cfg.TRAIN.GAUSSIAN                                = EasyDict()
cfg.TRAIN.GAUSSIAN.DATASET                        = "GOOGLE_EARTH"
cfg.TRAIN.GAUSSIAN.BATCH_SIZE                     = 1
cfg.TRAIN.GAUSSIAN.EPS                            = 1e-8
cfg.TRAIN.GAUSSIAN.WEIGHT_DECAY                   = 0
cfg.TRAIN.GAUSSIAN.BETAS                          = (0.9, 0.999)
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_MODEL          = "vgg19"
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_LAYERS         = ["relu_3_1", "relu_4_1", "relu_5_1"]
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_WEIGHTS        = [0.125, 0.25, 1.0]
cfg.TRAIN.GAUSSIAN.N_EPOCHS                       = 500
cfg.TRAIN.GAUSSIAN.L1_LOSS_FACTOR                 = 10
cfg.TRAIN.GAUSSIAN.PERCEPTUAL_LOSS_FACTOR         = 10
cfg.TRAIN.GAUSSIAN.GAN_LOSS_FACTOR                = .5
cfg.TRAIN.GAUSSIAN.CKPT_SAVE_FREQ                 = 25
cfg.TRAIN.GAUSSIAN.GENERATOR                      = EasyDict()
cfg.TRAIN.GAUSSIAN.GENERATOR.LR                   = 1e-4
cfg.TRAIN.GAUSSIAN.DISCRIMINATOR                  = EasyDict()
cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.ENABLED          = True
cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.LR               = 1e-5
cfg.TRAIN.GAUSSIAN.DISCRIMINATOR.N_WARMUP_ITERS   = 100000



#
# Test
#
cfg.TEST                                          = EasyDict()
cfg.TEST.GAUSSIAN                                 = EasyDict()
cfg.TEST.GAUSSIAN.DATASET                         = "GOOGLE_EARTH"
cfg.TEST.GAUSSIAN.TEST_FREQ                       = 1
# fmt: on
