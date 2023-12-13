# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-12-13 15:32:08
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-12-13 17:05:14
# @Email:  root@haozhexie.com

import matplotlib.pyplot as plt
import os
import pickle
import sys
import torch

PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
sys.path.append(PROJECT_HOME)

from extensions.topdown_projector import TopDownProjectorFunction


with open(os.path.join(PROJECT_HOME, "data", "VOLUME_3D.pkl"), "rb") as fp:
    volume = pickle.load(fp)["vol"]
    volume = torch.from_numpy(volume).unsqueeze(dim=0).cuda().int()

seg_maps, height_fields = TopDownProjectorFunction.apply(volume)

plt.rcParams["figure.figsize"] = (36, 36)
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
ax1.imshow(seg_maps.squeeze().cpu().numpy())
ax2.imshow(height_fields.squeeze().cpu().numpy())
plt.savefig(os.path.join(PROJECT_HOME, "output", "debug.jpg"))
