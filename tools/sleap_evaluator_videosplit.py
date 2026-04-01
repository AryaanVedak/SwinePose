import h5py
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Config ────────────────────────────────────────────────────────────────
PRED_SLP  = "/path/to/sleap_predictions.slp"  # UPDATE: path to your SLEAP predictions file
COCO_DIR  = Path("/path/to/SwinePose")  # UPDATE: path to your SwinePose dataset root
SIGMA     = 0.072
NUM_KP    = 15

# ---- Step 1: Load image_id lookup from val_videosplit.json ---------------
with open(COCO_DIR / "annotations/val_videosplit.json") as f:
    val_data = json.load(f)

frame_to_imgid = {
    (img["video"], img["frame_idx"]): img["id"]
    for img in val_data["images"]
}
print(f"Loaded {len(frame_to_imgid)} val frames from val_videosplit.json")

# ---- Step 2: Convert SLEAP predictions to COCO format -------------------
with h5py.File(PRED_SLP, "r") as f:
    frames    = f["frames"][:]
    instances = f["instances"][:]
    points    = f["pred_points"][:]

    raw_videos = f["videos_json"][:]
    video_map  = {}
    for i, entry in enumerate(raw_videos):
        v = json.loads(entry.decode("utf-8"))
        video_map[i] = v["filename"]

coco_predictions = []
skipped = 0
matched = 0

for frame in frames:
    video_idx     = int(frame["video"])
    frame_idx     = int(frame["frame_idx"])
    inst_id_start = int(frame["instance_id_start"])
    inst_id_end   = int(frame["instance_id_end"])

    if video_idx not in video_map:
        skipped += 1
        continue

    video_path = video_map[video_idx]
    key        = (Path(video_path).name, frame_idx)

    if key not in frame_to_imgid:
        skipped += 1
        continue

    image_id = frame_to_imgid[key]

    for inst_idx in range(inst_id_start, inst_id_end):
        inst     = instances[inst_idx]
        pt_start = int(inst["point_id_start"])
        pt_end   = int(inst["point_id_end"])
        inst_pts = points[pt_start:pt_end]

        kps_flat  = []
        kp_scores = []

        for pt in inst_pts:
            x, y  = float(pt["x"]), float(pt["y"])
            score = float(pt["score"])

            if np.isnan(x) or np.isnan(y):
                kps_flat.extend([0.0, 0.0, 0])
                kp_scores.append(0.0)
            else:
                kps_flat.extend([x, y, 2])
                kp_scores.append(score)

        visible_scores = [s for s in kp_scores if s > 0]
        instance_score = float(np.mean(visible_scores)) if visible_scores else 0.0

        coco_predictions.append({
            "image_id":    image_id,
            "category_id": 1,
            "keypoints":   kps_flat,
            "score":       instance_score
        })
        matched += 1

print(f"Converted {matched} predictions ({skipped} skipped)")

# Save predictions
pred_path = COCO_DIR / "sleap_predictions_videosplit.json"
with open(pred_path, "w") as f:
    json.dump(coco_predictions, f)
print(f"Saved predictions to {pred_path}")

# ---- Step 3: Evaluate with pycocotools ----------------------------------
SIGMAS = np.array([SIGMA] * NUM_KP)

coco_gt   = COCO(str(COCO_DIR / "annotations/val_videosplit.json"))
coco_dt   = coco_gt.loadRes(str(pred_path))
coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
coco_eval.params.kpt_oks_sigmas = SIGMAS
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# ---- Step 4: Print results -----------------------------------------------
print("\n" + "=" * 55)
print("SLEAP UNet — Video-level split (σ=0.072, 679 val frames)")
print("=" * 55)
print(f"  AP        (OKS>0.50:0.95): {coco_eval.stats[0]:.4f}")
print(f"  AP @0.50  (OKS>0.50):      {coco_eval.stats[1]:.4f}")
print(f"  AP @0.75  (OKS>0.75):      {coco_eval.stats[2]:.4f}")
print(f"  AR        (OKS>0.50:0.95): {coco_eval.stats[5]:.4f}")
print(f"  AR @0.50:                  {coco_eval.stats[6]:.4f}")
print(f"  AR @0.75:                  {coco_eval.stats[7]:.4f}")