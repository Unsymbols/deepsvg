# Unsymbols
## CHANGES to deepsvg  
- SVG `view_box` now is an envelope over objects
    - The existing implementation didn't work for potrace SVG because translation didn't work
- Many other small changes I did to get it to run, new packages etc. etc. etc.
- pyproject, uv 
- sampling — allow sampling letters without providing an initialization vector 
- Jupyter notebooks to generate symbols and show them prettily

## Running

To run: python 3.9
```bash
uv pip install -e .
```

## Docker
- Preprocessing
    - Runs if DSVG_DO_PREPROCESS set
    - DSVG_PRP_DATA_FOLDER (in), DSVG_PRP_OUTPUT_FOLDER (out)
        - metadata assumed in PRP_OUTPUT_FOLDER/meta.csv
- Training
:xa

    - DSVG_TRAIN_LOGDIR (=tensorboard logs, model checkpoints)
    - DSVG_TRAIN_DATA_DIR (=folder with preprocessed dataset)
    - DSVG_TRAIN_RESUME if set to `--resume` will resume training
    - Hyperparameters:
        - DSVG_TRAIN_BS  (=whatever configs.unsymbols.env_ours_hier_org.py)
        - DSVG_TRAIN_NUM_GPUS (=2)
        - DSVG_TRAIN_DATA_DIR
        - DSVG_TRAIN_BS
        - DSVG_TRAIN_LR (then later multiplied by ngpus)
        - DSVG_TRAIN_WARMUP_STEPS
        - DSVG_TRAIN_CKPT_VAL_EVERY (both checkpoint and val)


### Building etc.
```bash
docker build -t gitlab.hs-anhalt.de:5050/unsymbols/training/deepsvg:latest -t gitlab.hs-anhalt.de:5050/unsymbols/training/deepsvg:0.1  .
```

### Running locally as example
```bash
Assuming your 1-svg dataset is available locally at /tmp/data2/datasets/myoutdir/1-svg (sorry), then:

❯ docker run -i -e DSVG_DO_PREPROCESS=1 -e DSVG_PRP_DATA_FOLDER=/data/data_folder -e DSVG_PRP_OUTPUT_FOLDER=/data/dsvout -e DSVG_TRAIN_LOGDIR=/data/logdir -e DSVG_TRAIN_BS=2 -e DSVG_TRAIN_DATA_DIR=/data/dsvout -e DSVG_TRAIN_GPUS=4 -v /tmp/data2/datasets/myoutdir/1-svg/:/data/data_folder  gitlab.hs-anhalt.de:5050/unsymbols/training/deepsvg:latest
```


## Old fixes

"No kernel available with this device" fixable by:
```bash
❯ uv pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
(for my "| NVIDIA-SMI 535.247.01             Driver Version: 535.247.01   CUDA Version: 12.2")
Got it from [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)

# Old README
(not all true and with many deletions by me, mostly for reference)

## Introduction

**Sponsored by <img src="https://www.lingosub.com/icon.svg" height=20 width=20 style="vertical-align: middle;"/> [LingoSub](https://www.lingosub.com): Learn languages by watching videos with AI-powered translations**

**and <img src="https://www.thumbnailspro.com/icon.svg" height=20 width=20 style="vertical-align: middle;"/> [ThumbnailsPro](https://www.thumbnailspro.com): Instant AI-generated Thumbnails, for videos that get clicks.**

This is the official code for the paper "DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation".
Please refer to section [below](#citation) for Citation details.

- Paper: [arXiv](https://arxiv.org/abs/2007.11301)
- Code: [GitHub](https://github.com/alexandre01/deepsvg)
- Project page: [link](https://alexandre01.github.io/deepsvg)
- Reshot AI: [link](https://www.reshot.ai)
- BricksAR: [link](https://www.bricksar.com)
- LingoSub: [link](https://www.lingosub.com)
- ClipLaunch: [link](https://www.cliplaunch.com)
- Featured on AI Unfolded:

<p align="center">
    <a href="https://www.aiunfolded.co">
        <img alt="AI Unfolded" src="https://www.aiunfolded.co/featured.svg" height=75>
    </a>
</p>

- 1min video:

[![DeepSVG video](docs/imgs/thumbnail.jpg)](https://youtu.be/w9Bu4u-SsKQ)

------------------------------------------------------------------------------------------------------------------------

## Installation

Start by cloning this Git repository:
```
git clone https://github.com/alexandre01/deepsvg.git
cd deepsvg
```

Create a new [conda](https://docs.conda.io/en/latest/miniconda.html) environment (Python 3.7):
```
conda create -n deepsvg python=3.7
conda activate deepsvg
```

And install the dependencies:
```
pip install -r requirements.txt
```

Please refer to [cairosvg](https://cairosvg.org/documentation/#installation)'s documentation for additional requirements of CairoSVG.
For example:
- on Ubuntu: `sudo apt-get install libcairo2-dev`.
- on macOS: `brew install cairo libffi`.

## Tested environments
- Ubuntu 18.04, CUDA 10.1
- macOS 10.13.6, CUDA 10.1, PyTorch installed from source


## Dataset
![icons_dataset](docs/imgs/icons_dataset.png)
Download the dataset using the script provided in `dataset/` by running:
```
cd dataset/
bash download.sh
```

If this is not working for you, download the dataset manually from Google Drive, place the files in the `dataset` folder, and unzip (this may take a few minutes).
- `icons_meta.csv` (9 MB): https://drive.google.com/file/d/10Zx4TB1-BEdWv1GbwcSUl2-uRFiqgUP1/view?usp=sharing
- `icons_tensor.zip` (3 GB): https://drive.google.com/file/d/1gTuO3k98u_Y1rvpSbJFbqgCf6AJi2qIA/view?usp=sharing

By default, the dataset will be saved with the following tree structure:
```
deepsvg
 └─dataset/
    ├── icons_meta.csv
    └── icons_tensor/
```

> **NOTE**: The `icons_tensor/` folder contains the 100k icons in pre-augmented PyTorch tensor format, which enables to easily reproduce our work.
For full flexbility and more research freedom, we however recommend downloading the original SVG icons from [icons8](https://icons8.com), for which you will need a paid plan.
Instructions to download the dataset from source are coming soon.

To download the Font-dataset, we recommend following SVG-VAE's instructions: https://github.com/magenta/magenta/tree/master/magenta/models/svg_vae.
For demo purposes, we also release a mini version of the dataset. To download it, run:
```
cd dataset/
bash download_fonts.sh
```

Or use these links:
- `fonts_meta.csv` (6 MB): https://drive.google.com/file/d/1PEukDlZ6IkEhh9XfTTMMtFOwdXOC3iKn/view?usp=sharing
- `fonts_tensor.zip` (92 MB): https://drive.google.com/file/d/15xPf2FrXaHZ0bf6htZzc9ORTMGHYz9DX/view?usp=sharing


## Dataloader
To process a custom dataset of SVGs, use the `SVGDataset` dataloader.
To preprocess them on the fly, you can set `already_preprocessed` to `False`, but we recommend preprocessing them before training for better I/O performance.

To do so, use the `dataset/preprocess.py` script:
```shell script
python -m dataset.preprocess --data_folder dataset/svgs/ --output_folder dataset/svgs_simplified/ --output_meta_file dataset/svg_meta.csv
```

This will preprocess all input svgs in a multi-threaded way and generate a meta data file, for easier training filtering.

Then modify the training configuration by providing the correct dataloader module, data folder and meta data file:

``` python
cfg.dataloader_module = "deepsvg.svg_dataset"
cfg.data_dir = "./dataset/svgs_simplified/"
cfg.meta_filepath = "./dataset/svg_meta.csv"
```

## Deep learning SVG library
DeepSVG has been developed along with a library for deep learning with SVG data. The main features are:
- Parsing of SVG files.
- Conversion of basic shapes and commands to the subset `m`/`l`/`c`/`z`.
- Path simplification, using Ramer-Douglas-Peucker and Philip J. Schneider algorithms.
- Data augmentation: translation, scaling, rotation of SVGs.
- Conversion to PyTorch tensor format.
- Draw utilities, including visualization of control points and exporting to GIF animations.

The notebook `notebooks/svglib.ipynb` provides a walk-trough of the `deepsvg.svglib` library. Here's a little sample code showing the flexibility of our library:
```python
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point, Angle

icon = SVG.load_svg("docs/imgs/dolphin.svg").normalize()
icon.simplify_heuristic()                                 # path simplifcation
icon.zoom(0.75).translate(Point(0, 5)).rotate(Angle(15))  # scale, translate, rotate
icon.draw()
```
![dolphin_png](docs/imgs/dolphin.png)

And making a GIF of the SVG is as easy as:
```python
icon.animate()
```
![dolphin_animate](docs/imgs/dolphin_animate.gif)

## Training

Start a training by running the following command.

```
python -m deepsvg.train --config-module configs.deepsvg.hierarchical_ordered
```

The (optional) `--log-dir` argument lets you choose the directory where model checkpoints and tensorboard logs will be saved.

## Inference (interpolations)

Download pretrained models by running:
```
cd pretrained/
bash download.sh
```

If this doesn't work, you can download them manually from Google Drive and place them in the `pretrained` folder.
- `hierarchical_ordered.pth.tar` (41 MB): https://drive.google.com/file/d/1tsVx_cnFunSf5vvPWPVTjZ84IQC2pIDm/view?usp=sharing
- `hierarchical_ordered_fonts.pth.tar` (41 MB): https://drive.google.com/file/d/11KBUWfexw3LDvSFOVxy072_VCFYKm3L-/view?usp=sharing


We provide sample code in `notebooks/interpolate.ipynb` to perform interpolation between pairs of SVG icons
and `notebooks/latent_ops.ipynb` for word2vec-like operations in the SVG latent space, as shown in the experiments of our paper.

