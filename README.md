# Self‑Supervised Flood Detection from Satellite Images

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Project Overview
This repository implements a **self‑supervised learning (SSL)** pipeline for flood detection using high‑resolution satellite imagery.  
The approach learns robust visual representations from unlabeled Sentinel‑2 / Landsat imagery and fine‑tunes a lightweight classifier on a small set of annotated flood masks. The resulting model can be deployed for rapid, large‑scale flood mapping in disaster response scenarios.

## Key Features
- **Self‑supervised pre‑training** with contrastive learning (MoCo v2) on raw satellite tiles.
- **Modular architecture** built on PyTorch, torchvision, and `timm` (for backbone selection).
- **Geospatial I/O** handled by `rasterio` for seamless raster reading/writing.
- **Scalable data pipeline** using `torch.utils.data.DataLoader` with on‑the‑fly augmentations.
- **Evaluation utilities** (IoU, F1, precision/recall) powered by `scikit-learn`.
- **Visualization tools** for overlaying predictions on true RGB composites via `matplotlib`.
- **Config‑driven experiments** (YAML) for reproducibility.

## Dataset
The pipeline expects a directory layout similar to:

```
data/
├── raw/
│   ├── images/          # .tif files (multi‑band Sentinel‑2/Landsat)
│   └── masks/           # binary flood masks (same spatial resolution)
└── splits/
    ├── train.txt        # one file path per line
    ├── val.txt
    └── test.txt
```

- **Images**: 3‑band (RGB) or 13‑band Sentinel‑2 tiles (uint16).  
- **Masks**: Single‑channel uint8 (0 = background, 1 = flood).  

If you do not have a local copy, you can download the publicly available **Sen1Floods11** dataset (or any other flood mapping benchmark) and reorganize it to match the structure above.

## Methodology
1. **Self‑Supervised Pre‑Training**  
   - Backbone: ResNet‑50 (or any `timm` model).  
   - Objective: MoCo v2 contrastive loss with random geo‑augmentations (random crop, flip, rotation, color jitter).  
   - Pre‑training runs for 200 epochs on the unlabeled image pool.

2. **Supervised Fine‑Tuning**  
   - Replace the projection head with a binary segmentation head (1×1 conv + up‑sampling).  
   - Loss: Binary Cross‑Entropy + Dice loss.  
   - Fine‑tune for 50 epochs on the labeled subset.

3. **Inference**  
   - Sliding‑window inference with overlap to handle large tiles.  
   - Post‑processing: small object removal, morphological closing.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/self-supervised-flood-detection.git
cd self-supervised-flood-detection
```

### 2. Create a conda environment (recommended)
```bash
conda create -n flooddet python=3.10 -y
conda activate flooddet
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt` includes** (all installable via `pip`):
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
rasterio>=1.3.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
opencv-python>=4.8.0
```

### 4. Verify installation
```bash
python -c "import torch, torchvision, timm, rasterio, sklearn, matplotlib; print('All imports OK')"
```

## Quick Start

### Prepare data
```bash
# Example: download a small subset for testing
bash scripts/download_sample_data.sh
```

### Run the full pipeline (pre‑train + fine‑tune + evaluate)
```bash
python run.py \
  --config configs/experiment.yaml \
  --mode full
```

- `--mode` can be `pretrain`, `finetune`, `eval`, or `full`.
- All hyper‑parameters are defined in the YAML config (see next section).

## Training

### Self‑Supervised Pre‑Training
```bash
python run.py \
  --config configs/pretrain.yaml \
  --mode pretrain
```

Key config fields:
- `model.backbone`: e.g., `resnet50`, `vit_base_patch16_224`.
- `ssl.moco.queue_size`: 65536
- `ssl.moco.momentum`: 0.999
- `training.epochs`: 200
- `training.batch_size`: 128
- `training.lr`: 0.03
- `training.num_workers`: 8

### Supervised Fine‑Tuning
```bash
python run.py \
  --config configs/finetune.yaml \
  --mode finetune
```

Key config fields:
- `model.backbone`: must match the pre‑trained checkpoint.
- `training.epochs`: 50
- `training.lr`: 1e-4
- `loss.dice_weight`: 0.5
- `loss.bce_weight`: 0.5

Checkpoints are saved under `outputs/<experiment_name>/checkpoints/`.

## Evaluation
```bash
python run.py \
  --config configs/eval.yaml \
  --mode eval \
  --ckpt outputs/exp01/checkpoints/best_f1.pth
```

Metrics reported:
- Intersection‑over‑Union (IoU)
- F1‑Score
- Precision / Recall
- Mean Average Precision (mAP) at thresholds 0.5–0.9

Results are written to `outputs/<experiment_name>/metrics.json` and printed to stdout.

## Inference
```bash
python inference.py \
  --ckpt outputs/exp01/checkpoints/best_f1.pth \
  --input data/raw/images/scene_001.tif \
  --output predictions/scene_001_flood.tif \
  --tile-size 512 \
  --overlap 64
```

- The script automatically handles tiling, GPU batching, and writes a GeoTIFF mask aligned with the input raster.

## Visualization
```bash
python visualize.py \
  --image data/raw/images/scene_001.tif \
  --mask data/raw/masks/scene_001_mask.tif \
  --pred predictions/scene_001_flood.tif \
  --output viz/scene_001_overlay.png
```

The generated PNG shows the RGB background, ground‑truth flood (green), and model prediction (red) with a semi‑transparent overlay.

## Configuration
All experiments are driven by YAML files located in `configs/`.  
A minimal example (`configs/base.yaml`) looks like:

```yaml
experiment_name: exp01
seed: 42
device: cuda

model:
  backbone: resnet50
  pretrained: false   # set to true for ImageNet init, false for SSL init

training:
  epochs: 50
  batch_size: 32
  lr: 1e-4
  num_workers: 8
  weight_decay: 1e-5

data:
  root: data/
  train_list: data/splits/train.txt
  val_list: data/splits/val.txt
  test_list: data/splits/test.txt
  tile_size: 256
  augmentations:
    - RandomHorizontalFlip
    - RandomVerticalFlip
    - RandomCrop
    - ColorJitter

loss:
  bce_weight: 0.5
  dice_weight: 0.5

optimizer:
  name: AdamW
  lr: 1e-4
  weight_decay: 1e-5

scheduler:
  name: CosineAnnealingLR
  T_max: 50
```

You can inherit from `base.yaml` using the `!include` tag (supported by the custom loader in `utils/config.py`).

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Ensure code style with `black` and `flake8`.
4. Add unit tests under `tests/` and run `pytest`.
5. Submit a pull request with a clear description of changes.

Please follow the **PEP 8** style guide and keep the documentation up‑to‑date.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## Citation
If you use this code or the associated methodology in your research, please cite:

```bibtex
@inproceedings{yourname2024selfsupervised,
  title={Self‑Supervised Flood Detection from Satellite Images},
  author={Your Name and Coauthor, A.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  pages={1234--1243},
  doi={10.1109/CVPR.2024.00123}
}
```

--- 

*Happy coding and stay dry!*