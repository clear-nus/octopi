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
from transformers import CLIPImageProcessor
from transformers.utils import logging


class PropertyClassifierEvaluator:
    def evaluate(self, preds, labels):
        return self.get_correct_num(preds, labels)
    
    def get_correct_num(self, preds, labels):
        return (labels == torch.argmax(preds, dim=1)).sum().item()


def main(configs, exp_name, g, device):
    # data
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    train_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="train", flip_p=configs["flip_p"])
    val_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="val")
    test_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="test")
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)
    # models
    encoder = CLIPTactileEncoder(clip_model=configs["use_clip"]).to(device)
    classifier = CLIPClassifier(output_size=configs["output_size"]).to(device)
    if configs["prompt_learning"]:
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
    else:
        clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
    vificlip = ViFiCLIP(clip, freeze_text_encoder=True).to(device)
    if configs["prompt_learning"]:
        for name, param in vificlip.named_parameters():
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    # training
    evaluator = PropertyClassifierEvaluator()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_clip = torch.optim.AdamW(vificlip.parameters(), lr=configs["lr"])
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=configs["classifier_lr"])
    scheduler_clip = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_clip, T_max=len(train_loader) / configs["gradient_accumulation_steps"], eta_min=configs["lr"] / 100)
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, T_max=len(train_loader) / configs["gradient_accumulation_steps"], eta_min=configs["classifier_lr"] / 100)
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
            loss = (loss_fn(hardness_preds, hardness_labels) + loss_fn(roughness_preds, roughness_labels) + loss_fn(texture_preds, texture_labels)) / configs["gradient_accumulation_steps"]
            loss.backward()
            if (train_batch_step + 1) % configs["gradient_accumulation_steps"] == 0:
                optimizer_clip.step()
                optimizer_classifier.step()
                scheduler_clip.step()
                scheduler_classifier.step()
                optimizer_clip.zero_grad()
                optimizer_classifier.zero_grad()
            num_train_samples += batch_size
            total_train_hardness_correct += evaluator.evaluate(hardness_preds, hardness_labels)
            total_train_roughness_correct += evaluator.evaluate(roughness_preds, roughness_labels)
            total_train_texture_correct += evaluator.evaluate(texture_preds, texture_labels)
            combined_preds = torch.cat([torch.unsqueeze(torch.argmax(hardness_preds, dim=-1), dim=-1), torch.unsqueeze(torch.argmax(roughness_preds, dim=-1), dim=-1), torch.unsqueeze(torch.argmax(texture_preds, dim=-1), dim=-1)], dim=-1)
            combined_labels = torch.cat([torch.unsqueeze(hardness_labels, dim=-1), torch.unsqueeze(roughness_labels, dim=-1), torch.unsqueeze(texture_labels, dim=-1)], dim=-1)
            total_train_combined_correct += np.sum(np.all(combined_preds.cpu().detach().numpy() == combined_labels.cpu().detach().numpy(), axis=-1))
        # validation
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
                combined_preds = torch.cat([torch.unsqueeze(torch.argmax(hardness_preds, dim=-1), dim=-1), torch.unsqueeze(torch.argmax(roughness_preds, dim=-1), dim=-1), torch.unsqueeze(torch.argmax(texture_preds, dim=-1), dim=-1)], dim=-1)
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
                combined_preds = torch.cat([torch.unsqueeze(torch.argmax(hardness_preds, dim=-1), dim=-1), torch.unsqueeze(torch.argmax(roughness_preds, dim=-1), dim=-1), torch.unsqueeze(torch.argmax(texture_preds, dim=-1), dim=-1)], dim=-1)
                combined_labels = torch.cat([torch.unsqueeze(hardness_labels, dim=-1), torch.unsqueeze(roughness_labels, dim=-1), torch.unsqueeze(texture_labels, dim=-1)], dim=-1)
                total_test_combined_correct += np.sum(np.all(combined_preds.cpu().detach().numpy() == combined_labels.cpu().detach().numpy(), axis=-1))
        print(f"\nTRAIN epoch: {epoch+1} / {epochs}")
        print(f"TRAIN accuracies [hardness, roughness, texture, combined]: {total_train_hardness_correct / num_train_samples}, {total_train_roughness_correct / num_train_samples}, {total_train_texture_correct / num_train_samples}, {total_train_combined_correct / num_train_samples}")
        print(f"VAL accuracies [hardness, roughness, texture, combined]: {total_val_hardness_correct / num_val_samples}, {total_val_roughness_correct / num_val_samples}, {total_val_texture_correct / num_val_samples}, {total_val_combined_correct / num_val_samples}")
        print(f"TEST accuracies [hardness, roughness, texture, combined]: {total_test_hardness_correct / num_test_samples}, {total_test_roughness_correct / num_test_samples}, {total_test_texture_correct / num_test_samples}, {total_test_combined_correct / num_test_samples}")
        if total_val_combined_correct / num_val_samples > best_val_acc:
            print("Saving encoder...")
            best_val_acc = total_val_combined_correct / num_val_samples
            encoder.model.vision_model = vificlip.clip_model.vision_model
            torch.save(encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/encoder.pt")
            torch.save(classifier.state_dict(), f"{configs['exps_path']}/{exp_name}/classifier.pt")
            torch.save(vificlip.state_dict(), f"{configs['exps_path']}/{exp_name}/vificlip.pt")


if __name__ == "__main__":
    exp_type = f"train_clip"
    config_path = f'configs/{exp_type}_config.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
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