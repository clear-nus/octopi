import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import yaml
import os
import sys

# Ensure utils module can be found
sys.path.append(os.path.join(os.path.dirname(__file__)))

# get http proxies
os.environ.setdefault('HTTP_PROXY', "http://127.0.0.1:1087")
os.environ.setdefault('HTTPS_PROXY', "http://127.0.0.1:1087")

from utils.dataset import CLIPPropertyUniqueDataset
from utils.model import CLIPTactileEncoder
from utils.promptclip import PromptLearningCLIPModel
from transformers import CLIPImageProcessor
from torch.utils.data import DataLoader

def main():
    config_path = "configs/train_clip_config.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    
    device = "cuda:6"
    
    print("Loading image processor and dataset...")
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    
    val_dataset = CLIPPropertyUniqueDataset(
        image_processor=image_processor, 
        data_path=configs["data_dir"], 
        split_name="test"
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=configs["batch_size"], 
        shuffle=False,
    )
    
    print("Loading Encoder Model...")
    encoder = CLIPTactileEncoder(clip_model=configs["use_clip"]).to(device)
    
    # Load Stage 1 weights
    encoder_path = "exps/2026_03_18_04_11_39_train_clip_clip_seed_0/encoder.pt"
    if not os.path.exists(encoder_path):
        print(f"Cannot find encoder at {encoder_path}")
        return
        
    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    except RuntimeError:
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
        encoder.model.vision_model = clip.vision_model
        encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=True)
    encoder.eval()
    
    all_embeddings = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (tactile_frames, hardness, roughness, texture, _) in enumerate(val_loader):
            # tactile_frames is a list of tensors -> tactile_frames[0] has shape [batch, l, c, h, w]
            tactile_data = tactile_frames[0].to(device) 
            
            features = encoder(tactile_data)
            
            # Since sequence length l=5, we mean-pool over l to get a single vector per video
            features = features.mean(dim=1) # (batch, patch_embed_size)
            
            all_embeddings.append(features.cpu().numpy())
            
    all_embeddings = np.vstack(all_embeddings)
    
    # Extract labels parallel to dataset elements
    object_names = val_dataset.objects
    
    # Clean up names for the legend
    # E.g., 'physiclear_apple_0' -> 'apple'
    def clean_name(name):
        n = name.replace('physiclear_', '')
        # Remove trailing numbers like _0, _1 if present
        parts = n.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return n
        
    simplified_labels = [clean_name(name) for name in object_names]
    
    print(f"Running t-SNE on {len(all_embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    os.makedirs("assets", exist_ok=True)
    
    # Plot 1: Object Categories
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=simplified_labels,
        palette=sns.color_palette("hsv", len(set(simplified_labels))),
        legend="full",
        alpha=0.8
    )
    plt.title("t-SNE of Stage 1 Tactile Encoder Embeddings (Colored by Object)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("assets/tactile_tsne_objects.png", dpi=300)
    print("Saved object plot to assets/tactile_tsne_objects.png")
    
    # Extract property values from dataset
    all_hardness = []
    all_roughness = []
    all_texture = []
    for i in range(len(val_dataset)):
        _, h, r, t, _ = val_dataset.get_frames_and_label(i, None)
        all_hardness.append(h)
        all_roughness.append(r)
        all_texture.append(t)
        
    # Plot 2: Hardness Label
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=all_hardness,
        palette="viridis",
        legend="full",
        alpha=0.8
    )
    plt.title("t-SNE Colored by Hardness Label")
    plt.savefig("assets/tactile_tsne_hardness.png", dpi=300)
    print("Saved hardness plot to assets/tactile_tsne_hardness.png")

    # Plot 3: Roughness Label
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=all_roughness,
        palette="plasma",
        legend="full",
        alpha=0.8
    )
    plt.title("t-SNE Colored by Roughness Label")
    plt.savefig("assets/tactile_tsne_roughness.png", dpi=300)
    print("Saved roughness plot to assets/tactile_tsne_roughness.png")

    # Plot 4: Texture Label
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=all_texture,
        palette="magma",
        legend="full",
        alpha=0.8
    )
    plt.title("t-SNE Colored by Texture Label")
    plt.savefig("assets/tactile_tsne_texture.png", dpi=300)
    print("Saved texture plot to assets/tactile_tsne_texture.png")

if __name__ == "__main__":
    main()