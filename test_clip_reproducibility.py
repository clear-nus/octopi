import os 
import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from utils.dataset import *
from utils.model import *
from utils.promptclip import *
import random
import yaml
import sys
from transformers import CLIPImageProcessor

class PropertyClassifierEvaluator:
    def evaluate(self, preds, labels):
        return self.get_correct_num(preds, labels)
    
    def get_correct_num(self, preds, labels):
        return (labels == torch.argmax(preds, dim=1)).sum().item()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # Hardcoded path to the experiment we want to test
    EXP_PATH = "exps/train_clip_seed_1"
    CONFIG_PATH = "configs/train_clip_config.yaml"
    
    print(f"Testing reproducibility for: {EXP_PATH}")

    # Load base config
    with open(CONFIG_PATH, 'r') as file:
        configs = yaml.safe_load(file)
    
    # Override seed to 1 as requested
    configs["seed"] = 1
    configs["cuda"] = 1
    
    # Setup device
    device = f'cuda:{configs["cuda"]}'
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    np.random.seed(configs["seed"])
    random.seed(configs["seed"])
    g = torch.Generator()
    g.manual_seed(configs["seed"])

    # Data
    print("Loading dataset...")
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    test_dataset = CLIPPropertyUniqueDataset(image_processor=image_processor, data_path=configs["data_dir"], split_name="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Models
    print("Initializing models...")
    classifier = CLIPClassifier(output_size=configs["output_size"]).to(device)
    
    if configs["prompt_learning"]:
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
    else:
        clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
    
    vificlip = ViFiCLIP(clip, freeze_text_encoder=True).to(device)

    # Load Weights
    print("Loading weights...")
    vificlip_path = os.path.join(EXP_PATH, "vificlip.pt")
    classifier_path = os.path.join(EXP_PATH, "classifier.pt")
    
    if not os.path.exists(vificlip_path):
        print(f"Error: {vificlip_path} not found!")
        return
    if not os.path.exists(classifier_path):
        print(f"Error: {classifier_path} not found!")
        return

    vificlip.load_state_dict(torch.load(vificlip_path, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    vificlip.eval()
    classifier.eval()

    # Evaluation
    print("Starting evaluation...")
    evaluator = PropertyClassifierEvaluator()
    
    total_test_hardness_correct, total_test_roughness_correct, total_test_texture_correct, total_test_combined_correct = 0, 0, 0, 0
    num_test_samples = 0
    
    with torch.no_grad():
        for test_sample_step, batch in enumerate(tqdm.tqdm(test_loader)):
            objects_tactile_frames, hardness_labels, roughness_labels, texture_labels, all_indices = batch
            hardness_labels, roughness_labels, texture_labels = hardness_labels.to(device), roughness_labels.to(device), texture_labels.to(device)
            batch_size = objects_tactile_frames[0].shape[0]
            
            all_tactile_embeds = []
            for otf in objects_tactile_frames:
                video_features, _, _, _ = vificlip(otf.to(device), None, None, all_indices)
                all_tactile_embeds.append(video_features)
            
            all_tactile_embeds = torch.cat(all_tactile_embeds, dim=-1)
            hardness_preds, roughness_preds, texture_preds = classifier(all_tactile_embeds)
            
            num_test_samples += batch_size
            total_test_hardness_correct += evaluator.evaluate(hardness_preds, hardness_labels)
            total_test_roughness_correct += evaluator.evaluate(roughness_preds, roughness_labels)
            total_test_texture_correct += evaluator.evaluate(texture_preds, texture_labels)
            
            combined_preds = torch.cat([
                torch.unsqueeze(torch.argmax(hardness_preds, dim=-1), dim=-1), 
                torch.unsqueeze(torch.argmax(roughness_preds, dim=-1), dim=-1), 
                torch.unsqueeze(torch.argmax(texture_preds, dim=-1), dim=-1)
            ], dim=-1)
            combined_labels = torch.cat([
                torch.unsqueeze(hardness_labels, dim=-1), 
                torch.unsqueeze(roughness_labels, dim=-1), 
                torch.unsqueeze(texture_labels, dim=-1)
            ], dim=-1)
            
            total_test_combined_correct += np.sum(np.all(combined_preds.cpu().detach().numpy() == combined_labels.cpu().detach().numpy(), axis=-1))

    print("\nResults:")
    print(f"TEST accuracies [hardness, roughness, texture, combined]:")
    print(f"Hardness: {total_test_hardness_correct / num_test_samples:.4f}")
    print(f"Roughness: {total_test_roughness_correct / num_test_samples:.4f}")
    print(f"Texture: {total_test_texture_correct / num_test_samples:.4f}")
    print(f"Combined: {total_test_combined_correct / num_test_samples:.4f}")

if __name__ == "__main__":
    main()
