"""
Score the five baselines on the SwinePose animal-level test split (COCO keypoint AP),
with a uniform sigma of 0.072 and with the per-keypoint sigmas measured from two
annotators.

    python eval_pig_benchmark.py <run dir with predictions/> <test_pigsplit.json> <measured_oks_sigmas.json>

Writes <run dir>/results.json.
"""

import contextlib
import io
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if len(sys.argv) != 4:
    sys.exit(__doc__)
RUN, GT, SIGMAS = sys.argv[1:4]
MEASURED = json.load(open(SIGMAS))["sigmas"]
MODELS = ["unet", "resnet50", "hrnet", "rtmpose", "vitpose"]


def evaluate(pred, sigmas):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(GT)
        ev = COCOeval(gt, gt.loadRes(pred), "keypoints")
        ev.params.kpt_oks_sigmas = np.array(sigmas)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats


results = {}
print(f"{'model':10s} {'AP':>6s} {'AP50':>6s} {'AP75':>6s} {'AR':>6s} {'AP meas.σ':>10s}")
for m in MODELS:
    pred = f"{RUN}/predictions/{m}.keypoints.json"
    if not os.path.exists(pred) or not json.load(open(pred)):
        print(f"{m:10s} no predictions")
        continue
    s, t = evaluate(pred, [0.072] * 15), evaluate(pred, MEASURED)
    results[m] = {"AP": s[0], "AP50": s[1], "AP75": s[2], "AR": s[5], "AP_measured_sigma": t[0]}
    print(f"{m:10s} {s[0]:6.3f} {s[1]:6.3f} {s[2]:6.3f} {s[5]:6.3f} {t[0]:10.3f}")
json.dump(results, open(f"{RUN}/results.json", "w"), indent=1)
