# -*- coding: utf-8 -*-
#
# @File:   io.py
# @Author: Haozhe Xie
# @Date:   2019-08-02 10:22:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-05-31 16:52:55
# @Email:  root@haozhexie.com

import csv
import io
import json
import numpy as np
import os
import pickle
import sys

from PIL import Image

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

from config import cfg

sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)

# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
mc_client = None
if cfg.MEMCACHED.ENABLED:
    import mc

    mc_client = mc.MemcachedClient.GetInstance(
        cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG
    )


class IO:
    @classmethod
    def get(cls, file_path):
        if not os.path.exists(file_path):
            return None

        _, file_extension = os.path.splitext(file_path)
        if file_extension in [".png", ".jpg", ".jpeg"]:
            return cls._read_img(file_path)
        if file_extension in [".pkl"]:
            return cls._read_pkl(file_path)
        if file_extension in [".npy"]:
            return cls._read_npy(file_path)
        if file_extension in [".csv"]:
            return cls._read_csv(file_path)
        if file_extension in [".json"]:
            return cls._read_json(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_img(cls, file_path):
        if mc_client is None:
            img = Image.open(file_path)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            img = Image.open(io.BytesIO(np.frombuffer(buf, np.uint8)))

        return img

    @classmethod
    def _read_pkl(cls, file_path):
        if mc_client is None:
            with open(file_path, "rb") as f:
                pkl = pickle.load(f)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            pkl = pickle.loads(buf)

        return pkl

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        if mc_client is None:
            return np.load(file_path)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            buf_bytes = buf.tobytes()
            if not buf_bytes[:6] == b"\x93NUMPY":
                raise Exception("Invalid npy file format.")

            header_size = int.from_bytes(buf_bytes[8:10], byteorder="little")
            header = eval(buf_bytes[10 : header_size + 10])
            dtype = np.dtype(header["descr"])
            return np.frombuffer(buf[header_size + 10 :], dtype).reshape(
                header["shape"]
            )

    @classmethod
    def _read_csv(cls, file_path):
        with open(file_path) as f:
            reader = csv.DictReader(f)
            return {int(r["id"]): r for r in reader}

    @classmethod
    def _read_json(cls, file_path):
        with open(file_path, "r") as f:
            return json.load(f)
