# -*- coding: utf-8 -*-
#
# @File:   discriminator.py
# @Author: Haozhe Xie
# @Date:   2024-03-09 20:37:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-09 20:37:15
# @Email:  root@haozhexie.com

import torch


class Generator(torch.nn.Module):
    def __init__(self, cfg, n_classes):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.n_classes = n_classes
