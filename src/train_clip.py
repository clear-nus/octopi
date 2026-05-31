import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import json
import numpy as np
from utils.dataset import *
from utils.model import *
from utils.promptclip import *
import random
import yaml
from datetime import datetime
import sys
from transformers import CLIPImageProcessor, get_cosine_schedule_with_warmup
from transformers.utils import logging


import torch.nn.functional as F


def ordinal_predict(logits):
    return logits.argmax(dim=1)


def ordinal_loss(logits, labels, n_classes=3, smoothing=0.1, weight=None):
    """Label-smoothed cross-entropy with optional per-class weighting."""
    if smoothing > 0:
        n = logits.size(-1)
        with torch.no_grad():
            smooth = torch.full_like(logits, smoothing / (n - 1))
            smooth.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        log_prob = F.log_softmax(logits, dim=-1)
        per_sample = -(smooth * log_prob).sum(dim=-1)
        if weight is not None:
            w = weight[labels]
            return (per_sample * w).sum() / w.sum().clamp_min(1e-8)
        return per_sample.mean()
    return F.cross_entropy(logits, labels, weight=weight)


def compute_class_weights(objects, n_classes=3, device="cpu", mode="scaled", strength=1.0, properties=None):
    """Per-class weights computed from train objects only.

    mode="full" returns mean-1 inverse-frequency weights. mode="scaled" interpolates
    between uniform and inverse-frequency using alpha = 1 - 1/ratio, so balanced
    properties stay close to uniform.
    """
    from collections import Counter
    if mode not in {"scaled", "full"}:
        raise ValueError(f"Unknown class_balance_mode: {mode}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"class_balance_strength must be in [0, 1], got {strength}")
    if properties is None:
        properties = ["hardness", "roughness", "texture"]
    properties = set(properties)
    weights = {}
    for prop in ["hardness", "roughness", "texture"]:
        if prop not in properties:
            weights[prop] = None
            continue
        counts = Counter(RANKS[prop][o] for o in objects)
        counts = {c: counts.get(c, 0) for c in range(n_classes)}
        total = sum(counts.values())
        inv = torch.tensor(
            [total / (n_classes * max(counts[c], 1)) for c in range(n_classes)],
            dtype=torch.float32, device=device,
        )
        inv = inv / inv.mean()
        if mode == "full":
            target = inv
        else:
            ratio = max(counts.values()) / max(min(counts.values()), 1)
            alpha = 1.0 - 1.0 / ratio
            target = (1.0 - alpha) * torch.ones_like(inv) + alpha * inv
            target = target / target.mean()
        w = (1.0 - strength) * torch.ones_like(inv) + strength * target
        weights[prop] = w / w.mean()
    return weights


class EMA:
    """Exponential moving average over a fixed list of parameter tensors.

    Training proceeds on the raw weights; call apply_shadow() before eval/save and
    restore() afterward so checkpoints reflect the averaged (lower-variance) weights.
    """
    def __init__(self, params, decay):
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    def update(self):
        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                p.copy_(s)

    def restore(self):
        with torch.no_grad():
            for b, p in zip(self.backup, self.params):
                p.copy_(b)
        self.backup = None


class SWA:
    """Equal-weight average of end-of-epoch weight snapshots (SWA-style tail averaging).

    Averaging starts at start_epoch; update() is called once per epoch. apply_shadow()/
    restore() swap the running average in for eval/checkpoint, matching EMA's interface.
    LR is left to the existing cosine scheduler (no cyclic-LR phase), so this is tail
    weight-averaging rather than full SWA. The model has no BatchNorm (CLIP uses
    LayerNorm), so no running-stat recalibration is needed on the averaged weights.
    """
    def __init__(self, params, start_epoch):
        self.params = list(params)
        self.start_epoch = start_epoch
        self.avg = [p.detach().clone() for p in self.params]
        self.n = 0
        self.backup = None

    def update(self):
        self.n += 1
        with torch.no_grad():
            if self.n == 1:
                for a, p in zip(self.avg, self.params):
                    a.copy_(p.detach())
            else:
                for a, p in zip(self.avg, self.params):
                    a.add_(p.detach() - a, alpha=1.0 / self.n)

    def apply_shadow(self):
        self.backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for a, p in zip(self.avg, self.params):
                p.copy_(a)

    def restore(self):
        with torch.no_grad():
            for b, p in zip(self.backup, self.params):
                p.copy_(b)
        self.backup = None


def pairwise_ranking_loss(logits, labels, margin=1.0):
    """Margin ranking loss on softmax expected rank within a batch."""
    n_classes = logits.size(-1)
    ranks = torch.arange(n_classes, device=logits.device).float()
    expected = (F.softmax(logits, dim=-1) * ranks).sum(dim=-1)  # (B,)
    labels_f = labels.float()
    label_diff = labels_f.unsqueeze(0) - labels_f.unsqueeze(1)   # (B, B)
    score_diff = expected.unsqueeze(0) - expected.unsqueeze(1)    # (B, B)
    mask = (label_diff > 0).float()
    loss = torch.clamp(margin - score_diff, min=0.0) * mask
    n_pairs = mask.sum()
    return loss.sum() / n_pairs if n_pairs > 0 else loss.sum()


class PropertyClassifierEvaluator:
    def evaluate(self, preds, labels):
        return self.get_correct_num(preds, labels)

    def get_correct_num(self, preds, labels):
        return (labels == ordinal_predict(preds)).sum().item()


def main(configs, exp_name, g, device):
    # data
    try:
        image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    except Exception as e:
        print(f"Failed to load CLIP ImageProcessor: {e}. Ensure you have internet access or the model is cached/downloaded.")
        raise
    max_frames = configs.get("max_frames", 5)
    train_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="train", flip_p=configs["flip_p"], max_frames=max_frames,
        rotation_degrees=configs.get("rotation_degrees", 0), color_jitter=configs.get("color_jitter", 0.0), gaussian_blur=configs.get("gaussian_blur", False))
    val_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="val", max_frames=max_frames)
    test_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="test", max_frames=max_frames)
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)
    # models
    encoder = CLIPTactileEncoder(clip_model=configs["use_clip"]).to(device)
    classifier = CLIPClassifier(output_size=configs["output_size"], decoupled_heads=configs.get("decoupled_heads", False), decoupled_head_dim=configs.get("decoupled_head_dim", 128)).to(device)
    if configs["prompt_learning"]:
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
    else:
        clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
    fusion_layers = configs.get("fusion_layers", [-2])
    vificlip = ViFiCLIP(clip, freeze_text_encoder=True, fusion_layers=fusion_layers).to(device)
    unfreeze_last_n = configs.get("unfreeze_last_n_layers", 0)
    finetune_lr = configs.get("finetune_lr", 1e-5)
    if configs["prompt_learning"]:
        total_layers = vificlip.clip_model.config.vision_config.num_hidden_layers
        for name, param in vificlip.named_parameters():
            if "VPT" in name:
                param.requires_grad_(True)
            elif unfreeze_last_n > 0 and "vision_model.encoder.layers" in name:
                try:
                    layer_idx = int(name.split("vision_model.encoder.layers.")[1].split(".")[0])
                    param.requires_grad_(layer_idx >= total_layers - unfreeze_last_n)
                except (IndexError, ValueError):
                    param.requires_grad_(False)
            else:
                param.requires_grad_(False)
    if configs.get("freeze_clip", False):
        for param in vificlip.parameters():
            param.requires_grad_(False)
    # training
    evaluator = PropertyClassifierEvaluator()
    smoothing = configs.get("label_smoothing", 0.1)
    class_weights = {"hardness": None, "roughness": None, "texture": None}
    if configs.get("class_balanced_loss", False):
        class_balance_mode = configs.get("class_balance_mode", "scaled")
        class_balance_strength = configs.get("class_balance_strength", 1.0)
        class_balance_properties = configs.get("class_balance_properties", ["hardness", "roughness", "texture"])
        class_weights = compute_class_weights(
            train_dataset.objects,
            device=device,
            mode=class_balance_mode,
            strength=class_balance_strength,
            properties=class_balance_properties,
        )
        printable_weights = {
            k: None if v is None else [round(x, 3) for x in v.tolist()]
            for k, v in class_weights.items()
        }
        print(f"Class-balanced loss ON (mode={class_balance_mode}, strength={class_balance_strength}, properties={class_balance_properties}). Weights: "
              f"{printable_weights}")
    vpt_params = [p for n, p in vificlip.named_parameters() if "VPT" in n and p.requires_grad]
    finetune_params = [p for n, p in vificlip.named_parameters() if "VPT" not in n and p.requires_grad]
    optimizer_clip_groups = [{"params": vpt_params, "lr": configs["lr"]}]
    if finetune_params:
        optimizer_clip_groups.append({"params": finetune_params, "lr": finetune_lr})
    weight_decay = configs.get("weight_decay", 0.01)
    optimizer_clip = torch.optim.AdamW(optimizer_clip_groups, weight_decay=weight_decay)
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=configs["classifier_lr"], weight_decay=weight_decay)
    # Calculate total steps across all epochs
    total_steps = (len(train_loader) / configs["gradient_accumulation_steps"]) * configs["num_epochs"]
    warmup_steps = int(configs.get("warmup_ratio", 0.05) * total_steps) # Default to 5% warmup
    scheduler_clip = get_cosine_schedule_with_warmup(
        optimizer_clip, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scheduler_classifier = get_cosine_schedule_with_warmup(
        optimizer_classifier, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    ema_decay = configs.get("ema_decay", 0.0) or 0.0
    ema = None
    if ema_decay > 0:
        ema_params = vpt_params + finetune_params + list(classifier.parameters())
        ema = EMA(ema_params, ema_decay)
        print(f"EMA ON (decay={ema_decay}) over {len(ema_params)} param tensors")
    swa = None
    if configs.get("swa", False):
        swa_start_epoch = int(configs.get("swa_start_epoch", max(0, configs["num_epochs"] - 5)))
        swa_params = vpt_params + finetune_params + list(classifier.parameters())
        swa = SWA(swa_params, swa_start_epoch)
        print(f"SWA ON (start_epoch={swa_start_epoch}) over {len(swa_params)} param tensors")
        if ema is not None:
            print("WARNING: both EMA and SWA enabled; SWA takes precedence for eval/checkpoint.")
    best_val_acc = -1
    epochs = configs["num_epochs"]
    for epoch in tqdm.tqdm(range(epochs)):
        total_train_hardness_correct, total_train_roughness_correct, total_train_texture_correct, total_train_combined_correct = 0, 0, 0, 0
        num_train_samples = 0
        vificlip.train()
        classifier.train()
        for train_batch_step, batch in enumerate(t:=tqdm.tqdm(train_loader)):
            objects_tactile_frames, hardness_labels, roughness_labels, texture_labels, all_indices = batch
            hardness_labels, roughness_labels, texture_labels = hardness_labels.to(device), roughness_labels.to(device), texture_labels.to(device)
            batch_size = objects_tactile_frames[0].shape[0]
            all_tactile_embeds = []
            for otf in objects_tactile_frames:
                video_features, _, _, _ = vificlip(otf.to(device), None, None, all_indices)
                all_tactile_embeds.append(video_features) # [(batch_size, output_size)]
            all_tactile_embeds = torch.cat(all_tactile_embeds, dim=-1) # (batch_size, output_size)
            hardness_preds, roughness_preds, texture_preds = classifier(all_tactile_embeds)
            ce_loss = ordinal_loss(hardness_preds, hardness_labels, smoothing=smoothing, weight=class_weights["hardness"]) + ordinal_loss(roughness_preds, roughness_labels, smoothing=smoothing, weight=class_weights["roughness"]) + ordinal_loss(texture_preds, texture_labels, smoothing=smoothing, weight=class_weights["texture"])
            rank_w = configs.get("ranking_loss_weight", 0.0)
            if rank_w > 0:
                margin = configs.get("ranking_margin", 1.0)
                rank_loss = (
                    pairwise_ranking_loss(hardness_preds, hardness_labels, margin) +
                    pairwise_ranking_loss(roughness_preds, roughness_labels, margin) +
                    pairwise_ranking_loss(texture_preds, texture_labels, margin)
                )
                loss = (ce_loss + rank_w * rank_loss) / configs["gradient_accumulation_steps"]
            else:
                loss = ce_loss / configs["gradient_accumulation_steps"]
            loss.backward()
            if (train_batch_step + 1) % configs["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(vificlip.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer_clip.step()
                optimizer_classifier.step()
                scheduler_clip.step()
                scheduler_classifier.step()
                optimizer_clip.zero_grad()
                optimizer_classifier.zero_grad()
                if ema is not None:
                    ema.update()
            num_train_samples += batch_size
            total_train_hardness_correct += evaluator.evaluate(hardness_preds, hardness_labels)
            total_train_roughness_correct += evaluator.evaluate(roughness_preds, roughness_labels)
            total_train_texture_correct += evaluator.evaluate(texture_preds, texture_labels)
            combined_preds = torch.stack([ordinal_predict(hardness_preds), ordinal_predict(roughness_preds), ordinal_predict(texture_preds)], dim=-1)
            combined_labels = torch.cat([torch.unsqueeze(hardness_labels, dim=-1), torch.unsqueeze(roughness_labels, dim=-1), torch.unsqueeze(texture_labels, dim=-1)], dim=-1)
            total_train_combined_correct += np.sum(np.all(combined_preds.cpu().detach().numpy() == combined_labels.cpu().detach().numpy(), axis=-1))
        # end-of-epoch SWA snapshot (no-op until swa_start_epoch)
        if swa is not None and epoch >= swa.start_epoch:
            swa.update()
        # validation runs on the averaged weights when enabled (SWA takes precedence over
        # EMA), so checkpoints reflect the lower-variance average
        if swa is not None:
            avg = swa if swa.n > 0 else None
        else:
            avg = ema
        if avg is not None:
            avg.apply_shadow()
        vificlip.eval()
        classifier.eval()
        # total_val_correct = 0
        total_val_hardness_correct, total_val_roughness_correct, total_val_texture_correct, total_val_combined_correct = 0, 0, 0, 0
        num_val_samples = 0
        with torch.no_grad():
            for val_sample_step, batch in enumerate(t:=tqdm.tqdm(val_loader)):
                objects_tactile_frames, hardness_labels, roughness_labels, texture_labels, all_indices = batch
                hardness_labels, roughness_labels, texture_labels = hardness_labels.to(device), roughness_labels.to(device), texture_labels.to(device)
                batch_size = objects_tactile_frames[0].shape[0]
                all_tactile_embeds = []
                for otf in objects_tactile_frames:
                    video_features, _, _, _ = vificlip(otf.to(device), None, None, all_indices)
                    all_tactile_embeds.append(video_features) # [(batch_size, output_size), (batch_size, output_size)]
                all_tactile_embeds = torch.cat(all_tactile_embeds, dim=-1) # (batch_size, output_size * 2)
                hardness_preds, roughness_preds, texture_preds = classifier(all_tactile_embeds)
                num_val_samples += batch_size
                total_val_hardness_correct += evaluator.evaluate(hardness_preds, hardness_labels)
                total_val_roughness_correct += evaluator.evaluate(roughness_preds, roughness_labels)
                total_val_texture_correct += evaluator.evaluate(texture_preds, texture_labels)
                combined_preds = torch.stack([ordinal_predict(hardness_preds), ordinal_predict(roughness_preds), ordinal_predict(texture_preds)], dim=-1)
                combined_labels = torch.cat([torch.unsqueeze(hardness_labels, dim=-1), torch.unsqueeze(roughness_labels, dim=-1), torch.unsqueeze(texture_labels, dim=-1)], dim=-1)
                total_val_combined_correct += np.sum(np.all(combined_preds.cpu().detach().numpy() == combined_labels.cpu().detach().numpy(), axis=-1))
        total_test_hardness_correct, total_test_roughness_correct, total_test_texture_correct, total_test_combined_correct = 0, 0, 0, 0
        num_test_samples = 0
        with torch.no_grad():
            for test_sample_step, batch in enumerate(t:=tqdm.tqdm(test_loader)):
                objects_tactile_frames, hardness_labels, roughness_labels, texture_labels, all_indices = batch
                hardness_labels, roughness_labels, texture_labels = hardness_labels.to(device), roughness_labels.to(device), texture_labels.to(device)
                batch_size = objects_tactile_frames[0].shape[0]
                all_tactile_embeds = []
                for otf in objects_tactile_frames:
                    video_features, _, _, _ = vificlip(otf.to(device), None, None, all_indices)
                    all_tactile_embeds.append(video_features) # [(batch_size, output_size), (batch_size, output_size)]
                all_tactile_embeds = torch.cat(all_tactile_embeds, dim=-1) # (batch_size, output_size * 2)
                hardness_preds, roughness_preds, texture_preds = classifier(all_tactile_embeds)
                num_test_samples += batch_size
                total_test_hardness_correct += evaluator.evaluate(hardness_preds, hardness_labels)
                total_test_roughness_correct += evaluator.evaluate(roughness_preds, roughness_labels)
                total_test_texture_correct += evaluator.evaluate(texture_preds, texture_labels)
                combined_preds = torch.stack([ordinal_predict(hardness_preds), ordinal_predict(roughness_preds), ordinal_predict(texture_preds)], dim=-1)
                combined_labels = torch.cat([torch.unsqueeze(hardness_labels, dim=-1), torch.unsqueeze(roughness_labels, dim=-1), torch.unsqueeze(texture_labels, dim=-1)], dim=-1)
                total_test_combined_correct += np.sum(np.all(combined_preds.cpu().detach().numpy() == combined_labels.cpu().detach().numpy(), axis=-1))
        print(f"\nTRAIN epoch: {epoch+1} / {epochs}")
        print(f"TRAIN accuracies [hardness, roughness, texture, combined]: {total_train_hardness_correct / num_train_samples}, {total_train_roughness_correct / num_train_samples}, {total_train_texture_correct / num_train_samples}, {total_train_combined_correct / num_train_samples}")
        print(f"VAL accuracies [hardness, roughness, texture, combined]: {total_val_hardness_correct / num_val_samples}, {total_val_roughness_correct / num_val_samples}, {total_val_texture_correct / num_val_samples}, {total_val_combined_correct / num_val_samples}")
        print(f"TEST accuracies [hardness, roughness, texture, combined]: {total_test_hardness_correct / num_test_samples}, {total_test_roughness_correct / num_test_samples}, {total_test_texture_correct / num_test_samples}, {total_test_combined_correct / num_test_samples}")
        val_mean_acc = (total_val_hardness_correct + total_val_roughness_correct + total_val_texture_correct) / (3 * num_val_samples)
        if val_mean_acc > best_val_acc:
            print("Saving encoder...")
            best_val_acc = val_mean_acc
            encoder.model.vision_model = vificlip.clip_model.vision_model
            torch.save(encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/encoder.pt")
            torch.save(classifier.state_dict(), f"{configs['exps_path']}/{exp_name}/classifier.pt")
            torch.save(vificlip.state_dict(), f"{configs['exps_path']}/{exp_name}/vificlip.pt")
        if avg is not None:
            avg.restore()


if __name__ == "__main__":
    exp_type = f"train_clip"
    config_path = f'configs/{exp_type}_config.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    if "EXP_ID" in os.environ:
        exp_id = os.environ["EXP_ID"]
    else:
        exp_id = input("Identifier for experiment: ")
    if len(exp_id) == 0:
        exp_id = exp_type
    else:
        exp_id = exp_type + "_" + exp_id

    # make stats and weights folders
    now = datetime.now()
    exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = exp_name + "_" + exp_id
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_name}", exist_ok=True)
    with open(f"{configs['exps_path']}/{exp_name}/{exp_type}_config.yaml", 'w') as file:
        documents = yaml.dump(configs, file)
        file.close()

    # log outputs
    sys.stdout = open(f"{configs['exps_path']}/{exp_name}/log.txt", 'w')
    logging.set_verbosity_error()

    # seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    # torch.use_deterministic_algorithms(True)
    np.random.seed(configs["seed"])
    random.seed(configs["seed"])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])
    device = f'cuda:{configs["cuda"]}' # for inputs and model if not device_map

    print("Training CLIP...")
    main(configs, exp_name, g, device)
    print("\nCLIP trained!")
