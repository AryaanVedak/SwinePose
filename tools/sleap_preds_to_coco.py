"""
Convert SLEAP predictions (.slp) to a COCO keypoint results file for a given
ground-truth COCO file, using the same rules as sleap_evaluator_videosplit.py:
unplaced points -> [0, 0, 0], placed points -> v=2, instance score = mean score
of the placed points. Frames are matched on (video file name, frame index).

    python sleap_preds_to_coco.py <predictions.slp> <gt_coco.json> <out.json>

Run with SLEAP's bundled interpreter (needs sleap-io).
"""

import json
import os
import sys

import numpy as np
import sleap_io as sio

pred_path, gt_path, out_path = sys.argv[1:4]
gt = json.load(open(gt_path))
img_id = {(im["video"], im["frame_idx"]): im["id"] for im in gt["images"]}

labels = sio.load_slp(pred_path, open_videos=False)
out, unmatched = [], 0
for lf in labels.labeled_frames:
    key = (os.path.basename(lf.video.filename), lf.frame_idx)
    if key not in img_id:
        unmatched += 1
        continue
    for inst in lf.predicted_instances:
        pts = inst.points
        kps, scores = [], []
        for (x, y), s in zip(pts["xy"], pts["score"]):
            if np.isnan(x) or np.isnan(y):
                kps += [0.0, 0.0, 0]
            else:
                kps += [float(x), float(y), 2]
                scores.append(float(s))
        out.append({"image_id": img_id[key], "category_id": 1, "keypoints": kps,
                    "score": float(np.mean([s for s in scores if s > 0])) if any(s > 0 for s in scores) else 0.0})
json.dump(out, open(out_path, "w"))
print(f"{len(out)} predictions for {len({p['image_id'] for p in out})} of {len(gt['images'])} images "
      f"({unmatched} predicted frames not in the GT file)")
