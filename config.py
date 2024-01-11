# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-01-09 18:24:17
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
cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY              = ["hf", "seg"]
cfg.DATASETS.CITY_SAMPLE.N_REPEAT                = 1
cfg.DATASETS.CITY_SAMPLE.N_CLASSES               = 9
cfg.DATASETS.CITY_SAMPLE.N_MIN_PIXELS            = 64
cfg.DATASETS.CITY_SAMPLE.VOL_SIZE                = 1536
cfg.DATASETS.CITY_SAMPLE.MAX_HEIGHT              = 384
cfg.DATASETS.CITY_SAMPLE.N_CITIES                = 5
cfg.DATASETS.CITY_SAMPLE.N_VIEWS                 = 3000
cfg.DATASETS.CITY_SAMPLE.CITY_STYLES             = ["Day"]
cfg.DATASETS.CITY_SAMPLE_BUILDING                = EasyDict()
cfg.DATASETS.CITY_SAMPLE_BUILDING.PIN_MEMORY     = ["hf", "seg", "footprint_bboxes"]
cfg.DATASETS.CITY_SAMPLE_BUILDING.N_REPEAT       = 1
cfg.DATASETS.CITY_SAMPLE_BUILDING.N_MIN_PIXELS   = 64
cfg.DATASETS.CITY_SAMPLE_BUILDING.FACADE_CLS_ID  = 7
cfg.DATASETS.CITY_SAMPLE_BUILDING.ROOF_CLS_ID    = 8
cfg.DATASETS.CITY_SAMPLE_BUILDING.INS_ID_RANGE   = [100, 5000]
cfg.DATASETS.CITY_SAMPLE_BUILDING.VOL_SIZE       = 672

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
cfg.WANDB.PROJECT                                = "City-Gen-HD"
cfg.WANDB.ENTITY                                 = "haozhexie"
cfg.WANDB.MODE                                   = "online"
cfg.WANDB.RUN_ID                                 = None
cfg.WANDB.SYNC_TENSORBOARD                       = False

#
# Network
#
cfg.NETWORK                                      = EasyDict()
# GANCraft
cfg.NETWORK.GANCRAFT                             = EasyDict()
cfg.NETWORK.GANCRAFT.BUILDING_MODE               = False
cfg.NETWORK.GANCRAFT.N_CLASSES                   = cfg.DATASETS.CITY_SAMPLE.N_CLASSES
cfg.NETWORK.GANCRAFT.FACADE_CLS_ID               = cfg.DATASETS.CITY_SAMPLE_BUILDING.FACADE_CLS_ID
cfg.NETWORK.GANCRAFT.ROOF_CLS_ID                 = cfg.DATASETS.CITY_SAMPLE_BUILDING.ROOF_CLS_ID
cfg.NETWORK.GANCRAFT.STYLE_DIM                   = 256
cfg.NETWORK.GANCRAFT.N_SAMPLE_POINTS_PER_RAY     = 24
cfg.NETWORK.GANCRAFT.DIST_SCALE                  = 0.25
cfg.NETWORK.GANCRAFT.CENTER_OFFSET               = (cfg.DATASETS.CITY_SAMPLE.VOL_SIZE - cfg.DATASETS.CITY_SAMPLE_BUILDING.VOL_SIZE) / 2
cfg.NETWORK.GANCRAFT.NORMALIZE_DELIMETER         = ([cfg.DATASETS.CITY_SAMPLE_BUILDING.VOL_SIZE,] * 2 
                                                    if cfg.NETWORK.GANCRAFT.BUILDING_MODE 
                                                    else [cfg.DATASETS.CITY_SAMPLE.VOL_SIZE,] * 2) + [cfg.DATASETS.CITY_SAMPLE.MAX_HEIGHT]
cfg.NETWORK.GANCRAFT.ENCODER                     = "LOCAL"
cfg.NETWORK.GANCRAFT.ENCODER_OUT_DIM             = 64 if cfg.NETWORK.GANCRAFT.BUILDING_MODE else 32
cfg.NETWORK.GANCRAFT.GLOBAL_ENCODER_N_BLOCKS     = 6
cfg.NETWORK.GANCRAFT.LOCAL_ENCODER_NORM          = "GROUP_NORM"
cfg.NETWORK.GANCRAFT.SKY_POS_EMD_LEVEL_RAYDIR    = 5
cfg.NETWORK.GANCRAFT.SKY_POS_EMD_INCLUDE_RAYDIR  = True
cfg.NETWORK.GANCRAFT.POS_EMD                     = "SIN_COS"
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_FEATURES     = True
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_CORDS        = False
cfg.NETWORK.GANCRAFT.HASH_GRID_N_LEVELS          = 16
cfg.NETWORK.GANCRAFT.HASH_GRID_LEVEL_DIM         = 8
cfg.NETWORK.GANCRAFT.HASH_GRID_RESOLUTION        = (cfg.DATASETS.CITY_SAMPLE_BUILDING.VOL_SIZE 
                                                    if cfg.NETWORK.GANCRAFT.BUILDING_MODE 
                                                    else cfg.DATASETS.CITY_SAMPLE.VOL_SIZE)
cfg.NETWORK.GANCRAFT.SIN_COS_FREQ_BENDS          = 10
cfg.NETWORK.GANCRAFT.SKY_HIDDEN_DIM              = 256
cfg.NETWORK.GANCRAFT.SKY_OUT_DIM_COLOR           = 64
cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM           = 256
cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_SIGMA        = 1
cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_COLOR        = 64
cfg.NETWORK.GANCRAFT.DIS_N_CHANNEL_BASE          = 128

#
# Train
#
cfg.TRAIN                                        = EasyDict()
# GANCraft
cfg.TRAIN.GANCRAFT                               = EasyDict()
cfg.TRAIN.GANCRAFT.DATASET                       = "CITY_SAMPLE_BUILDING" if cfg.NETWORK.GANCRAFT.BUILDING_MODE else "CITY_SAMPLE"
cfg.TRAIN.GANCRAFT.N_EPOCHS                      = 500
cfg.TRAIN.GANCRAFT.CKPT_SAVE_FREQ                = 25
cfg.TRAIN.GANCRAFT.BATCH_SIZE                    = 1
cfg.TRAIN.GANCRAFT.LR_GENERATOR                  = 1e-4
cfg.TRAIN.GANCRAFT.LR_DISCRIMINATOR              = 1e-5
cfg.TRAIN.GANCRAFT.DISCRIMINATOR_N_WARMUP_ITERS  = 100000
cfg.TRAIN.GANCRAFT.EPS                           = 1e-7
cfg.TRAIN.GANCRAFT.WEIGHT_DECAY                  = 0
cfg.TRAIN.GANCRAFT.BETAS                         = (0., 0.999)
cfg.TRAIN.GANCRAFT.CROP_SIZE                     = (192, 192) if cfg.NETWORK.GANCRAFT.BUILDING_MODE else (128, 128)
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_MODEL         = "vgg19"
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_LAYERS        = ["relu_3_1", "relu_4_1", "relu_5_1"]
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_WEIGHTS       = [0.125, 0.25, 1.0]
cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR               = 10
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR        = 10
cfg.TRAIN.GANCRAFT.GAN_LOSS_FACTOR               = 0.5
cfg.TRAIN.GANCRAFT.EMA_ENABLED                   = False
cfg.TRAIN.GANCRAFT.EMA_RAMPUP                    = 0.05
cfg.TRAIN.GANCRAFT.EMA_N_RAMPUP_ITERS            = 10000

#
# Test
#
cfg.TEST                                         = EasyDict()
cfg.TEST.GANCRAFT                                = EasyDict()
# TODO
cfg.TEST.GANCRAFT.DATASET                        = "CITY_SAMPLE_BUILDING" if cfg.NETWORK.GANCRAFT.BUILDING_MODE else "CITY_SAMPLE"
cfg.TEST.GANCRAFT.CROP_SIZE                      = (480, 270)
# fmt: on
