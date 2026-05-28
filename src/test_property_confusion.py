"""Pool test-set predictions across the 5 k-fold CLIP checkpoints and report
per-property 3x3 confusion matrices + per-class precision/recall.

Test objects are identical across folds, so this aggregates ~5x38 predictions
to see *which classes* drive the hardness/roughness errors. Read-only; trains nothing.
"""
import glob
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from dataset import CLIPPropertyUniqueDataset  # noqa: E402
from model import CLIPClassifier, ViFiCLIP  # noqa: E402
from promptclip import PromptLearningCLIPModel  # noqa: E402

PROPS = ["hardness", "roughness", "texture"]


def seed_worker(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_models(configs, device):
    clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
    vificlip = ViFiCLIP(clip, freeze_text_encoder=True,
                        fusion_layers=configs.get("fusion_layers", [-2])).to(device)
    classifier = CLIPClassifier(output_size=configs["output_size"]).to(device)
    return vificlip, classifier


@torch.no_grad()
def collect_preds(vificlip, classifier, loader, device):
    vificlip.eval()
    classifier.eval()
    preds = {p: [] for p in PROPS}
    labels = {p: [] for p in PROPS}
    for batch in loader:
        otfs, h, r, t, all_indices = batch
        embeds = []
        for otf in otfs:
            vf, _, _, _ = vificlip(otf.to(device), None, None, all_indices)
            embeds.append(vf)
        embeds = torch.cat(embeds, dim=-1)
        hp, rp, tp = classifier(embeds)
        for p, pred, lab in zip(PROPS, [hp, rp, tp], [h, r, t]):
            preds[p].append(pred.argmax(dim=1).cpu().numpy())
            labels[p].append(lab.numpy())
    return ({p: np.concatenate(preds[p]) for p in PROPS},
            {p: np.concatenate(labels[p]) for p in PROPS})


def confusion(y_true, y_pred, n=3):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def print_report(prop, cm):
    n = cm.shape[0]
    print(f"\n=== {prop.upper()} ===  (rows=true, cols=pred)")
    header = "       " + "".join(f"pred{j:>4}" for j in range(n)) + "   recall"
    print(header)
    for i in range(n):
        row = cm[i]
        rec = row[i] / row.sum() if row.sum() else 0.0
        print(f"true{i:>2} " + "".join(f"{v:>8}" for v in row) + f"   {rec:.2f}")
    prec_line = "prec   "
    for j in range(n):
        col = cm[:, j]
        prec = cm[j, j] / col.sum() if col.sum() else 0.0
        prec_line += f"{prec:>8.2f}"
    print(prec_line)
    acc = np.trace(cm) / cm.sum()
    print(f"overall acc = {acc:.3f}  (support={cm.sum()})")


def main():
    device = "cuda:6"
    pattern = sys.argv[1] if len(sys.argv) > 1 else "exps/*_clip_kfold_0_f*"
    ckpt_dirs = sorted(glob.glob(pattern))
    if not ckpt_dirs:
        raise SystemExit("no k-fold checkpoints found")
    cfg = yaml.safe_load(open(f"{ckpt_dirs[0]}/train_clip_config.yaml"))
    image_processor = CLIPImageProcessor.from_pretrained(cfg["use_clip"])
    test_ds = CLIPPropertyUniqueDataset(image_processor=image_processor,
                                        data_path=cfg["data_dir"], split_name="test",
                                        max_frames=cfg.get("max_frames", 8))
    g = torch.Generator(); g.manual_seed(0)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)

    vificlip, classifier = build_models(cfg, device)

    totals = {p: np.zeros((3, 3), dtype=int) for p in PROPS}
    for d in ckpt_dirs:
        vificlip.load_state_dict(torch.load(f"{d}/vificlip.pt", map_location=device))
        classifier.load_state_dict(torch.load(f"{d}/classifier.pt", map_location=device))
        preds, labels = collect_preds(vificlip, classifier, loader, device)
        for p in PROPS:
            totals[p] += confusion(labels[p], preds[p])
        fold = d.split("_f")[-1]
        accs = {p: round(float((preds[p] == labels[p]).mean()), 3) for p in PROPS}
        print(f"[fold {fold}] test per-prop acc: {accs}")

    print("\n################ POOLED ACROSS 5 FOLDS ################")
    for p in PROPS:
        print_report(p, totals[p])


if __name__ == "__main__":
    main()
