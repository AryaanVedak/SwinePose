#!/bin/bash
# Train and test the five baselines on the SwinePose animal-level split.
#
#   tools/run_pig_benchmark.sh          full run  -> $OUT
#   tools/run_pig_benchmark.sh trial    1 epoch per model (checks the pipeline)
#
# Before running:
#   1. python tools/make_pig_split.py <SwinePose>/annotations
#   2. Copy configs/*.py into your MMPose installation's configs/ folder.
#   3. Build a SLEAP training package (train_pigsplit.pkg.slp, images embedded) and a
#      test labels file (test_pigsplit.slp) from the pig-split frames, and set the paths
#      in configs/unet/*.yaml.
#   4. Set the paths below. MMPose's --cfg-options cannot take paths containing spaces.
set -euo pipefail

DATA=/path/to/SwinePose            # contains annotations/ and images/
OUT=/path/to/pig_benchmark_runs     # checkpoints, logs, predictions, results
MM=/path/to/mmpose                  # MMPose 1.3.2 checkout (configs copied into configs/)
UNET_CFG=/path/to/SwinePose-repo/configs/unet
SLEAP_TEST=/path/to/test_pigsplit.slp
REPO=/path/to/SwinePose-repo
PY=python                           # Python with MMPose and pycocotools
SLEAP_PY=python                     # Python with sleap-nn and sleap-io

MODE=${1:-full}
[ "$MODE" = trial ] && OUT=${OUT}_trial
A=$DATA/annotations
mkdir -p "$OUT/predictions"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$OUT/timing.txt"; }

epochs() { if [ "$MODE" = trial ]; then echo "train_cfg.max_epochs=$2 train_cfg.val_interval=1"; else echo "train_cfg.max_epochs=$1"; fi; }
data() {
    echo train_dataloader.dataset.data_root=$DATA train_dataloader.dataset.ann_file=annotations/train_pigsplit_internal.json \
         val_dataloader.dataset.data_root=$DATA val_dataloader.dataset.ann_file=annotations/val_pigsplit_internal.json \
         val_evaluator.ann_file=$A/val_pigsplit_internal.json \
         test_dataloader.dataset.data_root=$DATA test_dataloader.dataset.ann_file=annotations/test_pigsplit.json \
         test_evaluator.ann_file=$A/test_pigsplit.json test_evaluator.outfile_prefix=$OUT/predictions/$1
}
best() { ls -t "$OUT/$1"/best_coco_AP_epoch_*.pth | head -1; }
train_mm() {
    local name=$1 cfg=$2 ep=$3 tep=$4; shift 4
    log "start train $name"
    (cd $MM && $PY tools/train.py configs/$cfg.py --work-dir "$OUT/$name" "$@" \
        --cfg-options $(data $name) $(epochs $ep $tep)) > "$OUT/$name.train.log" 2>&1
    log "end   train $name"
}
test_mm() {
    log "start test $1"
    (cd $MM && $PY tools/test.py configs/$2.py "$(best $3)" --work-dir "$OUT/$1/test" \
        --cfg-options $(data $1)) > "$OUT/$1.test.log" 2>&1
    log "end   test $1"
}

for m in resnet50 hrnet rtmpose; do
    train_mm $m pig_$m 100 1
    test_mm  $m pig_$m $m
done
train_mm vitpose_stage1 pig_vitpose_stage1 30 1
train_mm vitpose        pig_vitpose_stage2 100 2 --resume "$(best vitpose_stage1)"
test_mm  vitpose pig_vitpose_stage2 vitpose

for m in centroid centered_instance; do
    log "start train unet $m"
    sleap-nn train --config "$UNET_CFG/$m.yaml" trainer_config.ckpt_dir="$OUT/unet" \
        $([ "$MODE" = trial ] && echo trainer_config.max_epochs=1 || true) > "$OUT/unet_$m.train.log" 2>&1
    log "end   train unet $m"
done
log "start test unet"
sleap-nn track -i "$SLEAP_TEST" -m "$OUT/unet/pig.centroid" -m "$OUT/unet/pig.centered_instance" \
    --only_labeled_frames -b 4 -n 1 -o "$OUT/unet/test.predictions.slp" > "$OUT/unet.test.log" 2>&1
$SLEAP_PY "$REPO/tools/sleap_preds_to_coco.py" "$OUT/unet/test.predictions.slp" "$A/test_pigsplit.json" \
    "$OUT/predictions/unet.keypoints.json" >> "$OUT/unet.test.log" 2>&1
log "end   test unet"

$PY "$REPO/tools/eval_pig_benchmark.py" "$OUT" "$A/test_pigsplit.json" "$REPO/splits/measured_oks_sigmas.json" | tee "$OUT/results.txt"
log "done"
