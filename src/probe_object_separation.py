"""Probe whether a trained CLIP tactile encoder separates objects that share the
same (hardness, roughness, texture) tuple -- in particular the irreducible TEST
collision hairbrush_handle vs ice_block ((H,R,T)=(2,0,0)).

Read-only; trains nothing. For each test object it embeds every press through the
encoder (ViFiCLIP feat_mean), then for object pairs reports 1-NN leave-one-out
separability (cosine) plus centroid cosine similarity. Lower centroid-sim / higher
LOO-acc = better separated.

Usage:
  python src/probe_object_separation.py EXP_DIR [EXP_DIR2 ...] \
      [--data_dir DIR] [--pairs hairbrush_handle:ice_block,...]

If --data_dir is omitted, each exp's own train_clip_config.yaml data_dir is used.
Pass the SAME --data_dir for two encoders to compare them on identical presses.
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from transformers import CLIPImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from dataset import CLIPPropertyUniqueDataset  # noqa: E402
from model import ViFiCLIP  # noqa: E402
from promptclip import PromptLearningCLIPModel  # noqa: E402
from constants import HARDNESS_RANK, ROUGHNESS_RANK, TEXTURE_RANK  # noqa: E402

DEVICE = os.environ.get("PROBE_DEVICE", "cuda:0")
# The TEST collision we care about, plus a couple of well-separated controls.
DEFAULT_PAIRS = [
    ("physiclear_hairbrush_handle", "physiclear_ice_block"),   # (2,0,0) == (2,0,0) collision
    ("physiclear_hairbrush_handle", "physiclear_bath_towel"),  # control: different props
    ("physiclear_ice_block", "physiclear_microfiber_cloth"),   # control: different props
]


def tup(o):
    return (HARDNESS_RANK[o], ROUGHNESS_RANK[o], TEXTURE_RANK[o])


@torch.no_grad()
def embed_test_objects(vificlip, test_ds, device):
    """Return {object_name: np.array(n_presses, d)} of L2-normalised embeddings."""
    vificlip.eval()
    by_obj = {}
    for i in range(len(test_ds)):
        otf_list, _h, _r, _t, _aux, all_indices = test_ds.get_frames_and_label(i, None)
        embeds = []
        for otf in otf_list:
            vf, _, _, _ = vificlip(otf.unsqueeze(0).to(device), None, None, all_indices)
            embeds.append(vf)
        vf = torch.cat(embeds, dim=-1)            # (1, d)
        vf = torch.nn.functional.normalize(vf, dim=-1)
        obj = test_ds.objects[i]
        by_obj.setdefault(obj, []).append(vf.squeeze(0).cpu().numpy())
    return {k: np.stack(v) for k, v in by_obj.items()}


def loo_1nn(a, b):
    """1-NN leave-one-out accuracy separating two objects' press embeddings (cosine)."""
    X = np.concatenate([a, b], axis=0)
    y = np.array([0] * len(a) + [1] * len(b))
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)        # exclude self
    nn = sim.argmax(axis=1)
    pred = y[nn]
    acc = float((pred == y).mean())
    chance = max(len(a), len(b)) / (len(a) + len(b))
    return acc, chance, len(a), len(b)


def centroid_sim(a, b):
    ca = a.mean(0); ca /= (np.linalg.norm(ca) + 1e-8)
    cb = b.mean(0); cb /= (np.linalg.norm(cb) + 1e-8)
    def intra(x):
        c = x.mean(0); c /= (np.linalg.norm(c) + 1e-8)
        return float((x @ c).mean())
    return float(ca @ cb), intra(a), intra(b)


def report_all_pairs(by_obj):
    """1NN-LOO + centroid cosine for every test object pair, sorted hardest-first."""
    objs = sorted(by_obj)
    rows = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            acc, chance, na, nb = loo_1nn(by_obj[a], by_obj[b])
            csim, _, _ = centroid_sim(by_obj[a], by_obj[b])
            rows.append((acc, csim, a, b, na, nb))
    # hardest first: lowest LOO acc, then highest centroid cosine
    rows.sort(key=lambda r: (r[0], -r[1]))
    print("  --- all test pairs (hardest separated first) ---")
    n_confused = sum(1 for r in rows if r[0] < 1.0)
    for acc, csim, a, b, na, nb in rows:
        an, bn = a.replace("physiclear_", ""), b.replace("physiclear_", "")
        coll = " COLLISION" if tup(a) == tup(b) else ""
        mark = "  <<< CONFUSED" if acc < 1.0 else ""
        print(f"    {an}{tup(a)} / {bn}{tup(b)}: 1NN-LOO={acc:.2f} centroid_cos={csim:.3f}{coll}{mark}")
    print(f"  --> {n_confused}/{len(rows)} pairs confused (1NN-LOO<1.0); "
          f"min LOO acc = {rows[0][0]:.2f}")


def run_encoder(exp_dir, data_dir, pairs, device, all_pairs=False):
    cfg = yaml.safe_load(open(f"{exp_dir}/train_clip_config.yaml"))
    dd = data_dir or cfg["data_dir"]
    print(f"\n================ {os.path.basename(exp_dir)} ================")
    print(f"  data_dir={dd}  aux={cfg.get('aux_classifier_target')} w={cfg.get('aux_loss_weight')}")
    image_processor = CLIPImageProcessor.from_pretrained(cfg["use_clip"])
    test_ds = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=dd,
                                        split_name="test", max_frames=cfg.get("max_frames", 8))
    clip = PromptLearningCLIPModel.from_pretrained(cfg["use_clip"], cfg).to(device)
    vificlip = ViFiCLIP(clip, freeze_text_encoder=True).to(device)
    ckpt = f"{exp_dir}/vificlip.pt"
    vificlip.load_state_dict(torch.load(ckpt, map_location=device))
    by_obj = embed_test_objects(vificlip, test_ds, device)
    print(f"  embedded test objects: " + ", ".join(f"{k.replace('physiclear_','')}({len(v)})" for k, v in by_obj.items()))
    for a, b in pairs:
        if a not in by_obj or b not in by_obj:
            print(f"  [skip] {a}/{b} not both in test embeddings")
            continue
        acc, chance, na, nb = loo_1nn(by_obj[a], by_obj[b])
        csim, ia, ib = centroid_sim(by_obj[a], by_obj[b])
        an, bn = a.replace("physiclear_", ""), b.replace("physiclear_", "")
        flag = "  <-- (H,R,T) COLLISION" if tup(a) == tup(b) else ""
        print(f"  {an}{tup(a)} vs {bn}{tup(b)}: "
              f"1NN-LOO acc={acc:.2f} (chance={chance:.2f}, n={na}/{nb}) | "
              f"centroid_cos={csim:.3f} intra=({ia:.3f},{ib:.3f}){flag}")
    if all_pairs:
        report_all_pairs(by_obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--pairs", default=None,
                    help="comma list a:b (short names, physiclear_ prefix added)")
    ap.add_argument("--all_pairs", action="store_true",
                    help="also report 1NN-LOO/centroid for every test object pair")
    args = ap.parse_args()
    if args.pairs:
        pairs = []
        for p in args.pairs.split(","):
            a, b = p.split(":")
            pairs.append(("physiclear_" + a, "physiclear_" + b))
    else:
        pairs = DEFAULT_PAIRS
    device = DEVICE if torch.cuda.is_available() else "cpu"
    for d in args.exp_dirs:
        run_encoder(d.rstrip("/"), args.data_dir, pairs, device, all_pairs=args.all_pairs)


if __name__ == "__main__":
    main()
