# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 21:27:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-09-18 14:46:53
# @Email:  root@haozhexie.com


import argparse
import cv2
import importlib
import logging
import torch
import os
import sys

import core
import utils.distributed

from pprint import pprint
from datetime import datetime

# Fix deadlock in DataLoader
cv2.setNumThreads(0)


def get_args_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="exp_name",
        help="The name of the experiment",
        default="%s" % datetime.now(),
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_file",
        help="Path to the config.py file",
        default="config.py",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        help="The dataset name to train or test.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        help="The GPU device to use (e.g., 0,1,2,3).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--ckpt",
        dest="ckpt",
        help="Initialize the network from a pretrained model.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run",
        dest="run_id",
        help="The unique run ID for WandB",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test", dest="test", help="Test the network.", action="store_true"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="The rank ID of the GPU. Automatically assigned by torch.distributed.",
        default=os.getenv("LOCAL_RANK", 0),
    )
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Read the experimental config
    exec(compile(open(args.cfg_file, "rb").read(), args.cfg_file, "exec"))
    cfg = locals()["__C"]

    # Parse runtime arguments
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.exp_name is not None:
        cfg.CONST.EXP_NAME = args.exp_name
    if args.dataset is not None:
        cfg.CONST.DATASET = args.dataset
    if args.ckpt is not None:
        cfg.CONST.CKPT = args.ckpt
    if args.run_id is not None:
        cfg.WANDB.RUN_ID = args.run_id
    if args.run_id is not None and args.ckpt is None:
        raise Exception("No checkpoints")

    # Print the current config
    local_rank = args.local_rank
    if local_rank == 0:
        pprint(cfg)

    # Initialize the DDP environment
    if torch.cuda.is_available() and not args.test:
        utils.distributed.set_affinity(local_rank)
        utils.distributed.init_dist(local_rank)

    # Start train/test processes
    if not args.test:
        core.train(cfg)
    else:
        if "CKPT" not in cfg.CONST or not os.path.exists(cfg.CONST.CKPT):
            logging.error("Please specify the file path of checkpoint.")
            sys.exit(2)

        core.test(cfg)


if __name__ == "__main__":
    # References: https://stackoverflow.com/a/53553516/1841143
    importlib.reload(logging)
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    main()
