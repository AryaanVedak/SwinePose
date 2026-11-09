"""
Pig-level (animal-disjoint) train/test split for SwinePose v2.

Rule, applied per floor (recording group): among the pigs recorded on that
floor only, choose the set whose frame count is closest to 20% of the floor's
frames; ties are broken with a fixed seed. Pigs recorded on more than one floor
always go to training. The internal training/selection split then divides the
training frames 85/15 at random (seed 42), as for the video-level split.

Input : <annotations>/{train,val}_videosplit.json + split_manifest.json
Output: <annotations>/
            train_pigsplit.json, test_pigsplit.json,
            train_pigsplit_internal.json, val_pigsplit_internal.json,
            pig_split_manifest.json

    python make_pig_split.py <SwinePose/annotations folder>
"""

import copy
import sys
import itertools
import json
import random
from collections import defaultdict

ANN = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
SEED, TEST_FRAC, VAL_FRAC = 42, 0.20, 0.15
FLOORS = ["Durofarm1", "Ikadan1", "Sunds1"]

src = [json.load(open(f"{ANN}/{n}")) for n in ["train_videosplit.json", "val_videosplit.json"]]
categories = src[0]["categories"]
pig_of = json.load(open(f"{ANN}/split_manifest.json"))["pig_id"]

# One record per frame, in a fixed order (video, frame index).
frames = []
for j in src:
    ann = {a["image_id"]: a for a in j["annotations"]}
    for im in j["images"]:
        frames.append((im, ann[im["id"]]))
frames.sort(key=lambda f: (f[0]["video"], f[0]["frame_idx"]))

pig_frames, pig_floors, floor_frames = defaultdict(int), defaultdict(set), defaultdict(int)
for im, _ in frames:
    p, g = pig_of[im["video"]], im["video"].split("_")[0]
    pig_frames[p] += 1
    pig_floors[p].add(g)
    floor_frames[g] += 1

# ── Choose test pigs ─────────────────────────────────────────────────────
rng = random.Random(SEED)
test_pigs = set()
for g in FLOORS:
    eligible = sorted((p for p in pig_frames if pig_floors[p] == {g}), key=int)
    target = TEST_FRAC * floor_frames[g]
    combos = [c for r in range(1, len(eligible)) for c in itertools.combinations(eligible, r)]
    gap = lambda c: abs(sum(pig_frames[p] for p in c) - target)
    best = min(gap(c) for c in combos)
    test_pigs |= set(rng.choice(sorted(c for c in combos if gap(c) == best)))

train_frames = [f for f in frames if pig_of[f[0]["video"]] not in test_pigs]
test_frames = [f for f in frames if pig_of[f[0]["video"]] in test_pigs]
assert not {pig_of[f[0]["video"]] for f in train_frames} & test_pigs


def coco(frames_):
    """New, unique ids within the file (the two video-split files reuse ids)."""
    images, anns = [], []
    for i, (im, a) in enumerate(frames_, 1):
        im, a = copy.deepcopy(im), copy.deepcopy(a)
        im["id"] = a["image_id"] = a["id"] = i
        images.append(im)
        anns.append(a)
    return {"categories": categories, "images": images, "annotations": anns}


def save(obj, name):
    json.dump(obj, open(f"{ANN}/{name}", "w"))


train, test = coco(train_frames), coco(test_frames)
save(train, "train_pigsplit.json")
save(test, "test_pigsplit.json")

# ── Internal training / checkpoint-selection split of the training pigs ──
idx = list(range(len(train["images"])))
random.Random(SEED).shuffle(idx)
n_val = round(len(idx) * VAL_FRAC)
val_ids = {train["images"][i]["id"] for i in idx[:n_val]}
for name, keep in [("train_pigsplit_internal.json", lambda i: i not in val_ids),
                   ("val_pigsplit_internal.json", lambda i: i in val_ids)]:
    save({"categories": categories,
          "images": [im for im in train["images"] if keep(im["id"])],
          "annotations": [a for a in train["annotations"] if keep(a["image_id"])]}, name)

manifest = {
    "rule": "per floor, the single-floor pigs whose frame count is closest to 20% of the floor's frames "
            "(ties broken with seed 42); pigs recorded on more than one floor are kept in training",
    "seed": SEED,
    "test_pigs": sorted(test_pigs, key=int),
    "train_pigs": sorted(set(pig_frames) - test_pigs, key=int),
    "test_videos": sorted({im["video"] for im, _ in test_frames}),
    "train_videos": sorted({im["video"] for im, _ in train_frames}),
    "train_frames": len(train_frames), "test_frames": len(test_frames),
    "internal_train_frames": len(train_frames) - n_val, "internal_val_frames": n_val,
}
json.dump(manifest, open(f"{ANN}/pig_split_manifest.json", "w"), indent=2)

for g in FLOORS:
    t = sum(1 for im, _ in test_frames if im["video"].startswith(g))
    print(f"{g}: test pigs {sorted((p for p in test_pigs if g in pig_floors[p]), key=int)}, "
          f"{t}/{floor_frames[g]} frames ({100 * t / floor_frames[g]:.0f}%)")
print(f"train: {len(manifest['train_pigs'])} pigs, {len(manifest['train_videos'])} videos, {len(train_frames)} frames "
      f"(internal {manifest['internal_train_frames']} + {n_val}) | "
      f"test: {len(test_pigs)} pigs, {len(manifest['test_videos'])} videos, {len(test_frames)} frames")
