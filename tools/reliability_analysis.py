"""
Annotation reliability of SwinePose from the released files: agreement between the
released labels and the independent second annotation of 255 frames.

    python reliability_analysis.py <SwinePose folder> [out.json]

<SwinePose folder> contains annotations/{train,val}_videosplit.json and
reliability/second_annotator.json. Requires numpy, pandas and pingouin.

Computed per keypoint and for three groups (observed axial + right side,
estimated left side, all keypoints):
  - Cohen's kappa on whether each keypoint is annotated (in the field of view);
  - Euclidean distance between the two placements: mean, median, RMSE, 95th pct;
  - ICC(2,1) (two-way random, absolute agreement, single rater; pingouin
    "ICC(A,1)") with 95% CI, separately for x and y, on body-centred coordinates:
    per frame, the mean position of the keypoints placed by both annotators
    (averaged over the two annotators) is subtracted, and when keypoints are
    pooled each keypoint is centred on its own mean;
  - OKS sigma per keypoint, COCO convention: sigma_i^2 = E[d_i^2 / s^2], with d_i
    the distance between the second annotation and the released label and s^2 the
    released annotation's area (the bounding-box area, as used for OKS).
"""

import json
import sys

import numpy as np
import pandas as pd
import pingouin as pg

if len(sys.argv) < 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "reliability_results.json"
MM_PER_PX = 1400 / 720          # 1.4 m of corridor across 720 px
GROUPS = {"observed (axial + right)": [0, 1, 6, 2, 3, 4, 5, 7, 8],
          "estimated (left)": [9, 10, 11, 12, 13, 14],
          "all keypoints": list(range(15))}


def by_file(path):
    j = json.load(open(path))
    ims = {i["id"]: i for i in j["images"]}
    return {ims[a["image_id"]]["file_name"]: a for a in j["annotations"]}, j["categories"][0]["keypoints"]


released, names = {}, None
for f in ["train_videosplit.json", "val_videosplit.json"]:
    r, names = by_file(f"{ROOT}/annotations/{f}")
    released.update(r)
second, _ = by_file(f"{ROOT}/reliability/second_annotator.json")

files = sorted(second)
arr = lambda a: np.where(np.array(a["keypoints"], float).reshape(15, 3)[:, 2:] > 0,
                         np.array(a["keypoints"], float).reshape(15, 3)[:, :2], np.nan)
A1 = np.array([arr(released[f]) for f in files])      # released labels
A2 = np.array([arr(second[f]) for f in files])        # second annotator
scale = np.sqrt([released[f]["area"] for f in files])
placed1, placed2 = ~np.isnan(A1[..., 0]), ~np.isnan(A2[..., 0])
both = placed1 & placed2
centre = np.array([np.nanmean(np.concatenate([A1[f][both[f]], A2[f][both[f]]]), axis=0) for f in range(len(files))])
C1, C2 = A1 - centre[:, None, :], A2 - centre[:, None, :]


def cohen_kappa(a, b):
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return None if pe >= 1 else float((po - pe) / (1 - pe))


def icc21(v1, v2):
    n = len(v1)
    df = pd.DataFrame({"unit": np.tile(np.arange(n), 2), "rater": np.repeat([1, 2], n),
                       "value": np.concatenate([v1, v2])})
    row = pg.intraclass_corr(df, targets="unit", raters="rater", ratings="value").set_index("Type").loc["ICC(A,1)"]
    return {"icc": float(row["ICC"]), "ci95": [float(row["CI95"][0]), float(row["CI95"][1])]}


def summarise(idx):
    p1, p2 = placed1[:, idx].ravel(), placed2[:, idx].ravel()
    u1, u2 = [], []
    for i in idx:
        sel = both[:, i]
        mu = np.vstack([C1[sel, i], C2[sel, i]]).mean(0)
        u1.append(C1[sel, i] - mu)
        u2.append(C2[sel, i] - mu)
    u1, u2 = np.vstack(u1), np.vstack(u2)
    d = np.linalg.norm(u1 - u2, axis=1)       # the centring shift cancels: equals the image-space distance
    return {"slots": int(p1.size), "both_placed": int((p1 & p2).sum()),
            "only_released": int((p1 & ~p2).sum()), "only_second": int((~p1 & p2).sum()),
            "neither": int((~p1 & ~p2).sum()), "pct_agree_in_frame": float(100 * np.mean(p1 == p2)),
            "cohen_kappa": cohen_kappa(p1, p2),
            "mean_px": float(d.mean()), "median_px": float(np.median(d)),
            "rmse_px": float(np.sqrt(np.mean(d ** 2))), "p95_px": float(np.percentile(d, 95)),
            "mean_mm": float(d.mean() * MM_PER_PX),
            "icc_x": icc21(u1[:, 0], u2[:, 0]), "icc_y": icc21(u1[:, 1], u2[:, 1])}


results = {"frames": len(files),
           "groups": {g: summarise(idx) for g, idx in GROUPS.items()},
           "keypoints": {}}
for i, n in enumerate(names):
    k = summarise([i])
    sel = both[:, i]
    k["oks_sigma"] = float(np.sqrt(np.mean((np.linalg.norm(A1[sel, i] - A2[sel, i], axis=1) / scale[sel]) ** 2)))
    results["keypoints"][n] = k
json.dump(results, open(OUT, "w"), indent=1)

print(f"{results['frames']} frames")
for g, r in results["groups"].items():
    print(f"{g:26s} kappa {r['cohen_kappa']:.3f}  mean {r['mean_px']:.1f} px  median {r['median_px']:.1f}  "
          f"RMSE {r['rmse_px']:.1f}  ICC x {r['icc_x']['icc']:.3f}  ICC y {r['icc_y']['icc']:.3f}")
print("OKS sigma per keypoint:", {n: round(v["oks_sigma"], 3) for n, v in results["keypoints"].items()})
