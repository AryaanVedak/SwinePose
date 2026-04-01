# SwinePose

**A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis**

[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/19358700)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-red)](https://2026.acmmm.org)

## Overview

SwinePose is the first publicly available lateral-view benchmark dataset for markerless pig pose estimation and gait analysis. It comprises **3,778 annotated frames** from **85 video clips** recorded across three commercial pig farms, with **15 anatomically defined keypoints** per pig instance in COCO keypoint format.

**Paper:** SwinePose: A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis — ACM Multimedia 2026

**Dataset:** https://zenodo.org/records/19358700

---

## Benchmark Results

All models evaluated on 679 test frames (17 held-out videos) with uniform σ=0.072.

| Model | AP (0.50:0.95) | AP@0.50 | AP@0.75 | AR |
|-------|---------------|---------|---------|-----|
| SLEAP UNet | 0.951 | 0.989 | 0.965 | 0.966 |
| ResNet-50 | 0.955 | 0.990 | 0.961 | 0.970 |
| HRNet-W32 | 0.958 | 0.990 | 0.973 | 0.973 |
| RTMPose-m | 0.935 | 0.990 | 0.962 | 0.951 |
| ViTPose-S | 0.960 | 0.990 | 0.965 | 0.974 |

---

## Repository Structure

```
SwinePose/
├── configs/                        # MMPose training configs
│   ├── pig_vitpose_stage1.py       # ViTPose-S Stage 1 (frozen backbone)
│   ├── pig_vitpose_stage2.py       # ViTPose-S Stage 2 (full fine-tuning)
│   ├── pig_hrnet.py                # HRNet-W32
│   ├── pig_resnet50.py             # ResNet-50 SimpleBaseline
│   ├── pig_rtmpose.py              # RTMPose-m
│   └── pig_vitpose_grayscale.py    # ViTPose-S grayscale input
├── tools/
│   ├── per_keypoint_ap_all.py      # Per-keypoint AP evaluation
│   ├── sleap_evaluator_videosplit.py  # SLEAP evaluation on test split
│   └── verify_zenodo_upload.py    # Dataset integrity verification
├── splits/
│   └── split_manifest.json        # Exact video-to-split assignment (SEED=42)
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Download the dataset

Download the dataset from Zenodo:
```
https://zenodo.org/records/19358700
```

Extract and place at your preferred location. The dataset contains:
```
annotations/
    train_videosplit.json   # 68 training videos, 3,099 frames
    val_videosplit.json     # 17 test videos, 679 frames
    split_manifest.json     # Reproducible split manifest
images/
    *.jpg                   # 3,778 extracted frames
```

### 2. Install dependencies

For MMPose models (ResNet-50, HRNet-W32, RTMPose-m, ViTPose-S):
```bash
pip install mmpose==1.3.2 mmcv mmengine
```

For SLEAP:
```bash
pip install sleap
```

### 3. Train a model

Update `data_root` in the config to point to your dataset location, then:

```bash
cd ~/mmpose

# ViTPose-S (two-stage)
python tools/train.py configs/pig_vitpose_stage1.py \
    --work-dir /path/to/checkpoints/vitpose/stage1

python tools/train.py configs/pig_vitpose_stage2.py \
    --work-dir /path/to/checkpoints/vitpose/stage2 \
    --resume /path/to/checkpoints/vitpose/stage1/best_coco_AP_epoch_XX.pth

# HRNet-W32 (single stage)
python tools/train.py configs/pig_hrnet.py \
    --work-dir /path/to/checkpoints/hrnet

# ResNet-50 (single stage)
python tools/train.py configs/pig_resnet50.py \
    --work-dir /path/to/checkpoints/resnet50

# RTMPose-m (single stage)
python tools/train.py configs/pig_rtmpose.py \
    --work-dir /path/to/checkpoints/rtmpose
```

### 4. Evaluate a model

```bash
python tools/test.py configs/pig_hrnet.py \
    /path/to/checkpoints/hrnet/best_coco_AP_epoch_XX.pth
```

### 5. Per-keypoint AP

```bash
python tools/per_keypoint_ap_all.py
```

---

## Keypoint Schema

| ID | Name | Region |
|----|------|--------|
| 0 | nose | Trunk |
| 1 | front | Trunk (spinous tuber of shoulder blade) |
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

---

## Citation

If you use SwinePose in your research, please cite:

```bibtex
@inproceedings{vedak2026swinpose,
  title     = {SwinePose: A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis},
  author    = {Vedak, Aryaan Dilip and Lai, Forbes and Thorup, Vivi M. and Mahmoud, Marwa},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  publisher = {ACM},
}
```

---

## License

Code in this repository is released under the [MIT License](LICENSE).
The SwinePose dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Authors

- Aryaan Dilip Vedak — University of Glasgow
- Forbes Lai — University of Glasgow
- Vivi M. Thorup — Aarhus University
- Marwa Mahmoud — University of Glasgow
