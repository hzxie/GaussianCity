# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-09-23 20:50:53
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
cfg.DATASETS.GOOGLE_EARTH.N_REPEAT                = 1
cfg.DATASETS.GOOGLE_EARTH.N_CITIES                = 400
cfg.DATASETS.GOOGLE_EARTH.N_VIEWS                 = 60
cfg.DATASETS.GOOGLE_EARTH.TRAIN_N_INSTANCES       = None
cfg.DATASETS.GOOGLE_EARTH.TRAIN_INSTANCE_RANGE    = None
cfg.DATASETS.GOOGLE_EARTH.TRAIN_CROP_SIZE         = (448, 448)
cfg.DATASETS.GOOGLE_EARTH.TEST_N_INSTANCES        = None
cfg.DATASETS.GOOGLE_EARTH.TEST_INSTANCE_RANGE     = None
cfg.DATASETS.GOOGLE_EARTH.TEST_CROP_SIZE          = (720, 405)
cfg.DATASETS.GOOGLE_EARTH.TRAIN_MIN_PIXELS        = 64
cfg.DATASETS.GOOGLE_EARTH.TRAIN_MAX_POINTS        = 16384
## The following parameters should be the same as scripts/dataset_generator.py
cfg.DATASETS.GOOGLE_EARTH.CAM_K                   = [1528.1469407006614, 0, 480, 0, 1528.1469407006614, 270, 0, 0, 1]
cfg.DATASETS.GOOGLE_EARTH.SENSOR_SIZE             = (960, 540)
cfg.DATASETS.GOOGLE_EARTH.FLIP_UD                 = False
cfg.DATASETS.GOOGLE_EARTH.N_CLASSES               = 8
cfg.DATASETS.GOOGLE_EARTH.PROJ_SIZE               = 2048
cfg.DATASETS.GOOGLE_EARTH.BLDG_RANGE              = [100, 32768]
cfg.DATASETS.GOOGLE_EARTH.BLDG_FACADE_CLSID       = 2
cfg.DATASETS.GOOGLE_EARTH.BLDG_ROOF_CLSID         = 7
cfg.DATASETS.GOOGLE_EARTH.Z_SCALE_SPECIAL_CLASSES = {"ROAD": 1, "WATER": 5, "ZONE": 6}
cfg.DATASETS.GOOGLE_EARTH.MAP_SIZE                = 2048
cfg.DATASETS.GOOGLE_EARTH.SCALE                   = 1
# The KITTI-360 Dataset Config
cfg.DATASETS.KITTI_360                            = EasyDict()
cfg.DATASETS.KITTI_360.DIR                        = "./data/kitti-360/processed"
cfg.DATASETS.KITTI_360.PIN_MEMORY                 = ["Rt", "centers"]
cfg.DATASETS.KITTI_360.N_REPEAT                   = 1
cfg.DATASETS.KITTI_360.VIEW_INDEX_FILE            = "./data/kitti-360/views.json"
cfg.DATASETS.KITTI_360.TRAIN_N_INSTANCES          = None
cfg.DATASETS.KITTI_360.TRAIN_INSTANCE_RANGE       = None
cfg.DATASETS.KITTI_360.TRAIN_CROP_SIZE            = (448, 224)
cfg.DATASETS.KITTI_360.TEST_N_INSTANCES           = None
cfg.DATASETS.KITTI_360.TEST_INSTANCE_RANGE        = None
cfg.DATASETS.KITTI_360.TEST_CROP_SIZE             = (704, 376)
cfg.DATASETS.KITTI_360.TRAIN_MIN_PIXELS           = 64
cfg.DATASETS.KITTI_360.TRAIN_MAX_POINTS           = 16384
## The following parameters should be the same as scripts/dataset_generator.py
cfg.DATASETS.KITTI_360.CAM_K                      = [552.554261, 0, 682.049453, 0, 552.554261, 238.769549, 0, 0, 1]
cfg.DATASETS.KITTI_360.SENSOR_SIZE                = (1408, 376)
cfg.DATASETS.KITTI_360.FLIP_UD                    = True
cfg.DATASETS.KITTI_360.N_CLASSES                  = 8
cfg.DATASETS.KITTI_360.PROJ_SIZE                  = 2048
cfg.DATASETS.KITTI_360.BLDG_RANGE                 = [100, 10000]
cfg.DATASETS.KITTI_360.BLDG_FACADE_CLSID          = 2
cfg.DATASETS.KITTI_360.BLDG_ROOF_CLSID            = 7
cfg.DATASETS.KITTI_360.CAR_RANGE                  = [10000, 16384]
cfg.DATASETS.KITTI_360.CAR_CLSID                  = 3
cfg.DATASETS.KITTI_360.Z_SCALE_SPECIAL_CLASSES    = {"ROAD": 1, "ZONE": 6}
cfg.DATASETS.KITTI_360.MAP_SIZE                   = 0
cfg.DATASETS.KITTI_360.SCALE                      = 1

#
# Constants
#
cfg.CONST                                         = EasyDict()
cfg.CONST.EXP_NAME                                = ""
cfg.CONST.N_WORKERS                               = 8
cfg.CONST.DATASET                                 = "GOOGLE_EARTH"

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
cfg.NETWORK.GAUSSIAN.SCALE_FACTOR                 = 0.65
cfg.NETWORK.GAUSSIAN.ENCODER                      = "GLOBAL"     # Options: "GLOBAL", "LOCAL", None
cfg.NETWORK.GAUSSIAN.ENCODER_OUT_DIM              = 5            # Options: 5, 64, 3
cfg.NETWORK.GAUSSIAN.GLOBAL_ENCODER_N_BLOCKS      = 6
cfg.NETWORK.GAUSSIAN.POS_EMD                      = "HASH_GRID"  # Options: "HASH_GRID", "SIN_COS"
cfg.NETWORK.GAUSSIAN.HASH_GRID_N_LEVELS           = 16
cfg.NETWORK.GAUSSIAN.HASH_GRID_LEVEL_DIM          = 8
cfg.NETWORK.GAUSSIAN.SIN_COS_FREQ_BENDS           = 10
cfg.NETWORK.GAUSSIAN.Z_DIM                        = None         # Options: None, 256
cfg.NETWORK.GAUSSIAN.MLP_HIDDEN_DIM               = 512
cfg.NETWORK.GAUSSIAN.MLP_N_SHARED_LAYERS          = 1
cfg.NETWORK.GAUSSIAN.ATTR_FACTORS                 = {"rgb": 2}
cfg.NETWORK.GAUSSIAN.ATTR_N_LAYERS                = {"rgb": 1}
cfg.NETWORK.GAUSSIAN.DIS_N_CHANNEL_BASE           = 128
cfg.NETWORK.GAUSSIAN.PTV3                         = EasyDict()
cfg.NETWORK.GAUSSIAN.PTV3.ENABLED                 = True
cfg.NETWORK.GAUSSIAN.PTV3.ORDER                   = ("cord")
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
cfg.TEST.GAUSSIAN.TEST_FREQ                       = 1
# fmt: on
