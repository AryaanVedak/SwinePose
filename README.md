# SwinePose

**A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis**

[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://doi.org/10.5281/zenodo.19358699)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

SwinePose is a lateral-view dataset for markerless pig pose estimation and gait analysis. It comprises **3,814 annotated frames** from **85 videos** of **27 pigs**, recorded at a single research facility (Aarhus University, Research Centre Foulum, Denmark) on three slatted floor types (cast iron, plastic and concrete), with **15 anatomically defined keypoints** per pig in COCO keypoint format.

**Dataset (version 2.0):** https://doi.org/10.5281/zenodo.19358699

This repository contains the training configurations, split files and scripts used to build the benchmark splits, measure annotation reliability and train and evaluate the baselines.

---

## Baseline Results

COCO keypoint AP over OKS thresholds 0.50:0.05:0.95. "AP (measured σ)" uses the per-keypoint OKS sigmas measured from two annotators (`splits/measured_oks_sigmas.json`); all other columns use a uniform σ = 0.072.

**Animal-level split (benchmark):** 7 unseen test pigs, 15 videos, 655 frames.

| Model | AP | AP@0.50 | AP@0.75 | AR | AP (measured σ) |
|-------|-----|---------|---------|-----|-----------------|
| SLEAP U-Net | 0.912 | 0.968 | 0.913 | 0.953 | 0.858 |
| ResNet-50 | 0.931 | 0.975 | 0.932 | 0.959 | 0.792 |
| HRNet-W32 | 0.930 | 0.974 | 0.933 | 0.960 | 0.800 |
| RTMPose-m | 0.907 | 0.974 | 0.935 | 0.940 | 0.664 |
| ViTPose-S | 0.937 | 0.970 | 0.937 | 0.964 | 0.813 |

**Video-level split (secondary comparison):** 17 test videos, 714 frames; the test pigs also appear in training videos.

| Model | AP | AP@0.50 | AP@0.75 | AR | AP (measured σ) |
|-------|-----|---------|---------|-----|-----------------|
| SLEAP U-Net | 0.945 | 0.989 | 0.962 | 0.963 | 0.910 |
| ResNet-50 | 0.957 | 0.990 | 0.969 | 0.974 | 0.876 |
| HRNet-W32 | 0.961 | 0.990 | 0.971 | 0.976 | 0.874 |
| RTMPose-m | 0.929 | 0.989 | 0.958 | 0.949 | 0.712 |
| ViTPose-S | 0.963 | 0.990 | 0.973 | 0.975 | 0.884 |

Each model was trained once (seed 42). With seven test pigs, the animal-level scores depend in part on which pigs were selected.

---

## Repository Structure

```
SwinePose/
├── configs/
│   ├── pig_hrnet.py                  # HRNet-W32
│   ├── pig_resnet50.py               # ResNet-50 SimpleBaseline
│   ├── pig_rtmpose.py                # RTMPose-m
│   ├── pig_vitpose_stage1.py         # ViTPose-S stage 1 (frozen backbone)
│   ├── pig_vitpose_stage2.py         # ViTPose-S stage 2 (full fine-tuning)
│   └── unet/                         # SLEAP U-Net (centroid + centered-instance)
├── splits/
│   ├── pig_split_manifest.json       # Animal-level split: pigs, videos, rule and seed
│   ├── split_manifest.json           # Video-level split and the pig ID of each video
│   └── measured_oks_sigmas.json      # Per-keypoint OKS sigmas from two annotators
├── tools/
│   ├── make_pig_split.py             # Generates the animal-level split files
│   ├── reliability_analysis.py       # Annotation-reliability statistics and measured sigmas
│   ├── run_pig_benchmark.sh          # Trains and tests the five baselines (animal-level split)
│   ├── run_video_benchmark.sh        # Same for the video-level split
│   ├── eval_pig_benchmark.py         # Scores the predictions (σ = 0.072 and measured σ)
│   ├── sleap_preds_to_coco.py        # Converts SLEAP predictions to COCO results
│   └── sleap_evaluator_videosplit.py # Earlier SLEAP evaluation script (video-level split)
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Download the dataset

Download `SwinePose_v2.0.zip` from https://doi.org/10.5281/zenodo.19358699 and extract it:

```
SwinePose/
├── README.md
├── CHANGES_v1_to_v2.json             # Every added and corrected frame since version 1.0
├── annotations/
│   ├── train_pigsplit.json           # Animal-level training split: 20 pigs, 70 videos, 3,159 frames
│   ├── test_pigsplit.json            # Animal-level test split: 7 pigs, 15 videos, 655 frames
│   ├── train_pigsplit_internal.json  # Training part of the animal-level training split (2,685 frames)
│   ├── val_pigsplit_internal.json    # Checkpoint-selection part (474 frames)
│   ├── pig_split_manifest.json
│   ├── train_videosplit.json         # Video-level training split: 68 videos, 3,100 frames
│   ├── val_videosplit.json           # Video-level test split: 17 videos, 714 frames
│   ├── train_internal.json           # Training part of the video-level training split (2,635 frames)
│   ├── val_internal.json             # Checkpoint-selection part (465 frames)
│   └── split_manifest.json
├── images/                           # 3,814 frames (720 × 576)
└── reliability/
    └── second_annotator.json         # Independent second annotation of 255 frames
```

Use the animal-level split for benchmarking: no pig appears in both training and test.

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

### 3. Train and evaluate

To reproduce all five baselines, set the paths at the top of `tools/run_pig_benchmark.sh` (animal-level split) or `tools/run_video_benchmark.sh` (video-level split) and run it. The script header lists the preparation steps.

To train a single MMPose model, copy the files in `configs/` into your MMPose installation's `configs/` folder and set `data_root` in each config to your extracted `SwinePose/` folder. The configs train on the video-level files by default; the run scripts switch them to the animal-level files with `--cfg-options`.

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

# Test
python tools/test.py configs/pig_hrnet.py /path/to/checkpoints/hrnet/best_coco_AP_epoch_XX.pth
```

### 4. Annotation reliability

```bash
python tools/reliability_analysis.py /path/to/SwinePose   # needs numpy, pandas, pingouin
```

This reproduces the inter-annotator agreement (Cohen's κ 0.947, mean distance 7.9 px, ICC(2,1)) and the measured OKS sigmas from the released files.

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

The camera sees the pig's right side. Axial and right-side keypoints are placed from direct observation; left-side keypoints are estimated from anatomy and bilateral symmetry and are not flagged separately in the files.

---

## Citation

If you use SwinePose in your research, please cite:

```bibtex
@misc{vedak2026swinepose,
  title     = {SwinePose: A Lateral-View Benchmark Dataset for Pig Pose Estimation and Gait Analysis},
  author    = {Vedak, Aryaan Dilip and Thorup, Vivi M{\o}rk{\o}re and Mahmoud, Marwa},
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
- Vivi M. Thorup, Aarhus University
- Marwa Mahmoud, University of Glasgow
