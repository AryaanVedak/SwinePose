# SwinePose

**A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis**

[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://doi.org/10.5281/zenodo.19358699)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

SwinePose is a lateral-view dataset for markerless pig pose estimation and gait analysis. It comprises **3,778 annotated frames** from **85 videos** of **27 pigs**, recorded at a single research facility (Aarhus University, Research Centre Foulum, Denmark) on three slatted floor types (cast iron, plastic and concrete), with **15 anatomically defined keypoints** per pig in COCO keypoint format.

**Dataset:** https://doi.org/10.5281/zenodo.19358699

---

## Baseline Results

All models evaluated on 679 test frames (17 held-out videos) with uniform σ = 0.072.

| Model | AP (0.50:0.95) | AP@0.50 | AP@0.75 | AR |
|-------|---------------|---------|---------|-----|
| SLEAP U-Net | 0.951 | 0.989 | 0.965 | 0.966 |
| ResNet-50 | 0.955 | 0.990 | 0.961 | 0.970 |
| HRNet-W32 | 0.958 | 0.990 | 0.973 | 0.973 |
| RTMPose-m | 0.935 | 0.990 | 0.962 | 0.951 |
| ViTPose-S | 0.960 | 0.990 | 0.965 | 0.974 |

---

## Repository Structure

```
SwinePose/
├── configs/                          # MMPose training configs
│   ├── pig_hrnet.py                  # HRNet-W32
│   ├── pig_resnet50.py               # ResNet-50 SimpleBaseline
│   ├── pig_rtmpose.py                # RTMPose-m
│   ├── pig_vitpose_stage1.py         # ViTPose-S stage 1 (frozen backbone)
│   └── pig_vitpose_stage2.py         # ViTPose-S stage 2 (full fine-tuning)
├── tools/
│   └── sleap_evaluator_videosplit.py # SLEAP evaluation on the test split
├── splits/
│   ├── split_manifest.json           # Video-to-split assignment (SEED=42)
│   ├── train_internal.json           # Internal training split (2,634 frames)
│   └── val_internal.json             # Internal checkpoint-selection split (465 frames)
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Download the dataset

Download `SwinePose_v1.0.zip` from https://doi.org/10.5281/zenodo.19358699 and extract it. The data are in `SwinePose_Zenodo/dataset/`:

```
annotations/
    train_videosplit.json   # 68 training videos, 3,099 frames
    val_videosplit.json     # 17 test videos, 679 frames
    split_manifest.json     # Video-to-split assignment
images/
    *.jpg                   # 3,778 frames (720 × 576)
```

Copy `splits/train_internal.json` and `splits/val_internal.json` from this repository into `annotations/`. The MMPose configs train on these files.

### 2. Install dependencies

MMPose models (ResNet-50, HRNet-W32, RTMPose-m, ViTPose-S):
```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0,<2.2.0" "mmpose==1.3.2"
```

SLEAP U-Net:
```bash
pip install "sleap==1.6.1"
```

### 3. Train a model

Copy the files in `configs/` into your MMPose installation's `configs/` folder, set `data_root` in each config to your `SwinePose_Zenodo/dataset/` path, then:

```bash
cd /path/to/mmpose

# ViTPose-S (two stages)
python tools/train.py configs/pig_vitpose_stage1.py \
    --work-dir /path/to/checkpoints/vitpose/stage1
python tools/train.py configs/pig_vitpose_stage2.py \
    --work-dir /path/to/checkpoints/vitpose/stage2 \
    --resume /path/to/checkpoints/vitpose/stage1/best_coco_AP_epoch_XX.pth

# HRNet-W32, ResNet-50, RTMPose-m (single stage)
python tools/train.py configs/pig_hrnet.py    --work-dir /path/to/checkpoints/hrnet
python tools/train.py configs/pig_resnet50.py --work-dir /path/to/checkpoints/resnet50
python tools/train.py configs/pig_rtmpose.py  --work-dir /path/to/checkpoints/rtmpose
```

### 4. Evaluate a model

```bash
python tools/test.py configs/pig_hrnet.py \
    /path/to/checkpoints/hrnet/best_coco_AP_epoch_XX.pth
```

---

## Keypoint Schema

| ID | Name | Region |
|----|------|--------|
| 0 | nose | Trunk |
| 1 | front | Trunk (spinous tuber of the scapula) |
| 2 | right_carpal | Forelimb |
| 3 | right_elbow | Forelimb |
| 4 | right_tarsal | Hindlimb |
| 5 | right_stifle | Hindlimb |
| 6 | hip | Trunk |
| 7 | front_right_toe | Distal |
| 8 | rear_right_toe | Distal |
| 9 | left_carpal | Forelimb |
| 10 | left_elbow | Forelimb |
| 11 | front_left_toe | Distal |
| 12 | left_tarsal | Hindlimb |
| 13 | left_stifle | Hindlimb |
| 14 | rear_left_toe | Distal |

Left-side keypoints are not visible to the camera and were estimated from anatomy and bilateral symmetry.

---

## Citation

If you use SwinePose in your research, please cite:

```bibtex
@misc{vedak2026swinepose,
  title     = {SwinePose: A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis},
  author    = {Vedak, Aryaan Dilip and Lai, Forbes and Thorup, Vivi M{\o}rk{\o}re and Mahmoud, Marwa},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19358699}
}
```

---

## License

Code in this repository is released under the [MIT License](LICENSE).
The SwinePose dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Authors

- Aryaan Dilip Vedak, University of Glasgow
- Forbes Lai, University of Glasgow
- Vivi M. Thorup, Aarhus University
- Marwa Mahmoud, University of Glasgow
