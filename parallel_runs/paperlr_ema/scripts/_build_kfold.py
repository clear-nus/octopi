"""Build K stratified folds from TRAIN_OBJECTS ∪ VAL_OBJECTS.

Stratifies by HARDNESS_RANK (3-class), since that's the property the encoder
seems to fit on most cleanly. Writes one JSON per fold to scripts/kfold/fold_<i>.json:

    {"train_objects": [...], "val_objects": [...]}

TEST_OBJECTS is never touched.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "utils"))
import constants as c  # noqa: E402


def stratified_kfold(pool, k, get_class, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for obj in pool:
        buckets[get_class(obj)].append(obj)
    for cls in buckets:
        rng.shuffle(buckets[cls])
    folds = [[] for _ in range(k)]
    for cls, objs in buckets.items():
        for i, obj in enumerate(objs):
            folds[i % k].append(obj)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="scripts/kfold")
    args = ap.parse_args()

    pool = list(c.TRAIN_OBJECTS) + list(c.VAL_OBJECTS)
    missing = [o for o in pool if o not in c.HARDNESS_RANK]
    if missing:
        raise SystemExit(f"objects missing from HARDNESS_RANK: {missing}")

    folds = stratified_kfold(pool, args.k, lambda o: c.HARDNESS_RANK[o], args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, val_objs in enumerate(folds):
        train_objs = [o for o in pool if o not in set(val_objs)]
        out = {"train_objects": train_objs, "val_objects": sorted(val_objs)}
        with open(f"{args.out_dir}/fold_{i}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"fold {i}: train={len(train_objs)} val={len(val_objs)} "
              f"val_objs={sorted(val_objs)}")


if __name__ == "__main__":
    main()
