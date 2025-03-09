<img src="https://www.infinitescript.com/projects/GaussianCity/GaussianCity-Logo.webp" height="150px" align="right">

# Generative Gaussian Splatting for Unbounded 3D City Generation

[Haozhe Xie](https://haozhexie.com/about), [Zhaoxi Chen](https://frozenburning.github.io/), [Fangzhou Hong](https://hongfz16.github.io/), [Ziwei Liu](https://liuziwei7.github.io/)

S-Lab, Nanyang Technological University

[![codebeat badge](https://codebeat.co/badges/652ea895-6855-4488-a4f6-ba8d1e2f83a1)](https://codebeat.co/projects/github-com-hzxie-gaussiancity-master)
![Counter](https://api.infinitescript.com/badgen/count?name=hzxie/GaussianCity)
[![arXiv](https://img.shields.io/badge/arXiv-2406.06526-b31b1b.svg)](https://arxiv.org/abs/2406.06526)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/spaces/hzxie/gaussian-city)
[![YouTube](https://img.shields.io/badge/Spotlight%20Video-%23FF0000.svg?logo=YouTube&logoColor=white)](https://youtu.be/anDwIXlfjUA)

![Teaser](https://www.infinitescript.com/projects/GaussianCity/GaussianCity-Teaser.webp)

## Changelog 🔥

- [2025/03/02] The [Hugging Face demo](https://huggingface.co/spaces/hzxie/gaussian-city) is available (Ranked among the Top 8 Spaces in [Week 11, 2025](https://huggingface.co/spaces?date=2025-W11)).
- [2025/02/27] The training and testing code is released.
- [2024/05/24] The repo is created.

## Cite this work 📝

```
@inproceedings{xie2025gaussiancity,
  title     = {Generative Gaussian Splatting for Unbounded 3{D} City Generation},
  author    = {Xie, Haozhe and 
               Chen, Zhaoxi and 
               Hong, Fangzhou and 
               Liu, Ziwei},
  booktitle = {CVPR},
  year      = {2025}
}
```

## Datasets & Pretrained Models 🛢️

**Datasets**

- [OSM](https://gateway.infinitescript.com/s/OSM)
- [GoogleEarth](https://gateway.infinitescript.com/s/GoogleEarth)

**Pretrained Models**

- [BG-Generator.pth](https://gateway.infinitescript.com/?f=GaussianCity-REST-GoogleEarth.pth)
- [BLDG-Generator.pth](https://gateway.infinitescript.com/?f=GaussianCity-BLDG-GoogleEarth.pth)

## Installation 📥

Assume that you have installed [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org) in your Python (or Anaconda) environment.  

The GaussianCity source code is tested in PyTorch 2.4.1 with CUDA 11.8 in Python 3.11. You can use the following command to install PyTorch built on CUDA 11.8.

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

After that, the Python dependencies can be installed as following.

```bash
git clone https://github.com/hzxie/GaussianCity
cd GaussianCity
GCITY_HOME=`pwd`
pip install -r requirements.txt
```

The CUDA extensions can be compiled and installed with the following commands.

```bash
cd $GCITY_HOME/extensions
for e in `ls -d */`
do
  cd $GCITY_HOME/extensions/$e
  pip install .
done
```

## Inference 🚩

The command line interface (CLI) by default load the pretrained models for Background Generator and Building Generator from `output/rest.pth` and `output/bldg.pth`, respectively. You have the option to specify a different location using runtime arguments.

```
├── ...
└── GaussianCity
    └── scripts
    |   ├── ...
    |   └── inference.py
    └── output
        ├── bldg.pth
        └── rest.pth
```

Run the following command to generate 3D cities. The output video will be saved at `output/rendering.mp4`.

```bash
python3 scripts/inference.py
```

**Important Note:** The inference speed with `inference.py` is **NOT** 60 times faster than CityDreamer, as the `footprint_extruder` runs on the CPU for better compatibility. For faster inference, please use the GPU implementation available on [Hugging Face](https://huggingface.co/spaces/hzxie/gaussian-city/blob/main/gaussiancity/extensions/voxlib/points_to_volume.cu).

## Training👩🏽‍💻

### Dataset Preparation

By default, all scripts load the [OSM](https://gateway.infinitescript.com/s/OSM) and [GoogleEarth](https://gateway.infinitescript.com/s/GoogleEarth) datasets from `./data/osm` and `./data/google-earth`, respectively. You have the option to specify a different location using runtime arguments.

```
├── ...
└── GaussianCity
    └── data
        ├── google-earth
        └── osm 
```

1. Generate semantic segmentation using [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once), following the guidelines provided in [CityDreamer's README](https://github.com/hzxie/CityDreamer/tree/master?tab=readme-ov-file#dataset-preparation).
2. Generate instance segmentation with the following command.

```bash
cd $GCITY_HOME
python3 scripts/dataset_generator.py
```

### Training Background Generator

#### Update `config.py` ⚙️

Based on the default configuration file `config.py`, modify the settings as follows:

```python
cfg.DATASETS.GOOGLE_EARTH.TRAIN_N_INSTANCES       = 0
cfg.DATASETS.GOOGLE_EARTH.TRAIN_INSTANCE_RANGE    = [0, 10]
cfg.DATASETS.GOOGLE_EARTH.TRAIN_CROP_SIZE         = (640, 448)
cfg.NETWORK.GAUSSIAN.SCALE_FACTOR                 = 0.5
cfg.NETWORK.GAUSSIAN.PTV3.ENABLED                 = False
cfg.NETWORK.GAUSSIAN.PTV3.ORDER                   = ("z")
```

#### Launch Training 🚀

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py -e BG-Exp
```

### Training Building Generator

#### Update `config.py` ⚙️

Based on the default configuration file `config.py`, modify the settings as follows:

```python
cfg.DATASETS.GOOGLE_EARTH.TRAIN_N_INSTANCES       = 1
cfg.DATASETS.GOOGLE_EARTH.TRAIN_INSTANCE_RANGE    = [10, 16384]
cfg.DATASETS.GOOGLE_EARTH.TRAIN_CROP_SIZE         = (640, 448)
cfg.NETWORK.GAUSSIAN.SCALE_FACTOR                 = 0.5
cfg.NETWORK.GAUSSIAN.ENCODER                      = None
cfg.NETWORK.GAUSSIAN.ENCODER_OUT_DIM              = 3
cfg.NETWORK.GAUSSIAN.POS_EMD                      = "SIN_COS"
cfg.NETWORK.GAUSSIAN.Z_DIM                        = 256
```

#### Launch Training 🚀

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py -e BLDG-Exp
```

## License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/hzxie/GaussianCity/blob/master/LICENSE). Redistribution and use should follow this license.

