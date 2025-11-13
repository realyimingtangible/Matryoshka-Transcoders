import yaml
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel
import shutil
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from train_transcoder import *
from train_sae import *
from train_matryoshka_sae import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_TEXT_CHUNK = 100 * (1024 ** 2)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["general"]["cuda_devices"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =================== Matryoshka Transcoder Architecture ===================

class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()


def topk_mask(tensor, k):
    """Apply top-k sparsity mask to tensor"""
    if k is None or k >= tensor.shape[1]:
        return tensor

    topk_vals, topk_indices = torch.topk(tensor.abs(), k, dim=1)
    mask = torch.zeros_like(tensor, device=tensor.device)
    mask.scatter_(1, topk_indices, 1)
    return tensor * mask


class MatryoshkaTranscoder(nn.Module):
    def __init__(self, input_dim, output_dim, matryoshka_config, jump_params=None):
        super(MatryoshkaTranscoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.levels = matryoshka_config['levels']
        self.topk_per_level = matryoshka_config['topk_per_level']
        self.level_weights = matryoshka_config['level_weights']
        self.num_levels = len(self.levels)
        self.latent_dim = self.levels[-1]

        gamma, beta = jump_params if jump_params else (1.0, 1.0)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim, bias=True),
            JumpReLU(gamma=gamma, beta=beta)
        )

        self.decoders = nn.ModuleList([
            nn.Linear(level_dim, output_dim, bias=True)
            for level_dim in self.levels
        ])

    def apply_nested_topk(self, z):
        z_sparse = torch.zeros_like(z)
        level_starts = [0] + self.levels[:-1]
        level_ends = self.levels

        for i, (start, end, k) in enumerate(zip(level_starts, level_ends, self.topk_per_level)):
            level_features = z[:, start:end]
            level_features_sparse = topk_mask(level_features, k)
            z_sparse[:, start:end] = level_features_sparse

        return z_sparse

    def encode(self, h_2, level=None):
        """Encode inputs at specific granularity level"""
        z = self.encoder(h_2)
        z_sparse = self.apply_nested_topk(z)

        if level is not None:
            return z_sparse[:, :self.levels[level]]
        return z_sparse

    def get_activations(self, h_2):
        """Get sparse latent activations"""
        return self.encode(h_2)


# =================== Dataset (IMAGE-ONLY) ===================

class ImageOnlyDataset(Dataset):
    def __init__(self, dataset_path, exts=(".png", ".jpg", ".jpeg")):
        all_files = os.listdir(dataset_path)
        self.image_files = [f for f in all_files if f.lower().endswith(exts)]
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        return image, image_path


# =================== Classifier (for extracting h_2) ===================

class Classifier(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base = base_model

        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False

        # Classification head (kept but optional)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def get_base_features(self, image_features):
        """Extract h_2 (base CLIP features)"""
        return image_features

    def get_hidden_features(self, image_features):
        """Extract h_1 (hidden layer features)"""
        return self.relu(self.fc1(image_features))


# =================== Feature Extraction ===================

def load_models(model_num):
    """Load classifier and transcoder models"""

    # Load CLIP model
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        local_files_only=True
    ).to(device)

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        local_files_only=True
    )

    # Load trained classifier (if exists)
    classifier = Classifier(clip_model).to(device)
    classifier_path = f"clip_classifier.pt"
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print(f"✓ Loaded classifier from {classifier_path}")
    else:
        print(f"Warning: Classifier not found at {classifier_path} (continuing with base frozen CLIP)")

    # Load transcoder
    # transcoder_path = f"./models/matryoshka_transcoder_{model_num}_best.pt"
    # if not os.path.exists(transcoder_path):
    #     transcoder_path = f"./models/matryoshka_transcoder_{model_num}.pt"
    # transcoder_path = f"./models/sparse_autoencoder_best.pt"
    transcoder_path = f"./models/matryoshka_sae_{model_num}_best.pt"

    checkpoint = torch.load(transcoder_path, map_location=device)

    # Extract configuration
    transcoder_config = checkpoint['config']
    # matryoshka_config = {
    #     'levels': transcoder_config['levels'],
    #     'topk_per_level': transcoder_config['topk_per_level'],
    #     'level_weights': transcoder_config['level_weights']
    # }

    # # Initialize transcoder
    # transcoder = MatryoshkaTranscoder(
    #     input_dim=transcoder_config['input_dim'],
    #     output_dim=transcoder_config['output_dim'],
    #     matryoshka_config=matryoshka_config,
    #     jump_params=(1.0, 1.0)
    # ).to(device)

    #  # Initialize transcoder
    # transcoder = SimpleTranscoder(
    #     input_dim=transcoder_config['input_dim'],
    #     output_dim=transcoder_config['output_dim'],
    #     latent_dim=2048,
    #     jump_params=(1.0, 1.0)
    # ).to(device)

    # transcoder.load_state_dict(checkpoint['model_state_dict'])
    # transcoder.eval()

    # Matryoshka configuration (preserve)
    matryoshka_config = {
        'levels': [128, 256, 512, 1024, 2048],  # Cumulative latent dimensions
        'topk_per_level': [16, 32, 64, 128, 256],  # Top-k sparsity at each level
        'level_weights': [0.1, 0.15, 0.2, 0.25, 0.3]  # Loss weights (must sum to 1.0)
    }
    

    # Initialize transcoder
    model = MatryoshkaSparseAutoencoder(
        input_dim=transcoder_config['input_dim'],
        # output_dim=transcoder_config['output_dim'],
        matryoshka_config=matryoshka_config,
        jump_params=(1.0, 1.0)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded transcoder from {transcoder_path}")
    # print(f"  Levels: {transcoder.levels}")
    # print(f"  Top-k per level: {transcoder.topk_per_level}")

    return clip_model, clip_processor, classifier, model


def populate_features(model_num, all_folders, target_level=None,
                      activation_threshold=0.0, max_images_per_feature=20):
    """
    Extract and save top-activating images for each transcoder feature (IMAGE-ONLY)
    """
    torch.backends.cudnn.benchmark = True

    # Load models
    clip_model, clip_processor, classifier, transcoder = load_models(model_num)

    # Setup output directories
    OUTPUT_BASE = f"./transcoder_features_model{model_num}"

    if target_level is not None:
        OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"level_{target_level}")
        print(f"\nExtracting features from level {target_level}")
        print(f"  Dimension: {transcoder.levels[target_level]}")
        print(f"  Top-k: {transcoder.topk_per_level[target_level]}")
    else:
        OUTPUT_DIR = OUTPUT_BASE
        print(f"\nExtracting features from all levels")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 64
    base_path = "./datasets"

    # Track feature activations across all images
    feature_activations = defaultdict(list)  # feature_idx -> [(activation, img_path, dataset)]

    for dataset in all_folders:
        dataset_path = os.path.join(base_path, dataset)

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path not found: {dataset_path}")
            continue

        data_loader = DataLoader(
            ImageOnlyDataset(dataset_path),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: {
                "images": [item[0] for item in batch],
                "img_paths": [item[1] for item in batch]
            }
        )

        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 60}")

        for batch in tqdm(data_loader, desc=f"Extracting activations"):
            images = batch["images"]
            img_paths = batch["img_paths"]

            # Process images through CLIP
            X_inputs = clip_processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                # Get CLIP image features (h_2)
                img_features = clip_model.get_image_features(**X_inputs)

                # Get transcoder activations
                if target_level is not None:
                    z_sparse = transcoder.encode(img_features, level=target_level)
                    feature_offset = 0 if target_level == 0 else transcoder.levels[target_level - 1]
                else:
                    z_sparse = transcoder.get_activations(img_features)
                    feature_offset = 0

            # Move to CPU for processing
            z_sparse_cpu = z_sparse.cpu()

            # Record activations for each image
            for idx in range(z_sparse_cpu.shape[0]):
                activations = z_sparse_cpu[idx]

                # Find activated features (above threshold)
                activated_indices = (activations.abs() > activation_threshold).nonzero(as_tuple=True)[0]

                for local_feature_idx in activated_indices:
                    activation_value = activations[local_feature_idx].item()
                    global_feature_idx = feature_offset + local_feature_idx.item()

                    feature_activations[global_feature_idx].append((
                        abs(activation_value),
                        img_paths[idx],
                        dataset
                    ))

        print(f"✓ Processed {dataset}")

    # Save top-k images for each feature
    print(f"\n{'=' * 60}")
    print("Saving top-activating images for each feature")
    print(f"{'=' * 60}")

    num_features_with_activations = len(feature_activations)
    print(f"Found {num_features_with_activations} features with activations")

    for feature_idx, activations_list in tqdm(feature_activations.items(),
                                               desc="Saving features"):
        # Sort by activation strength (descending)
        activations_list.sort(key=lambda x: x[0], reverse=True)

        # Take top-k images
        top_activations = activations_list[:max_images_per_feature]

        # Create feature directory
        feature_dir = os.path.join(OUTPUT_DIR, f"feature_{feature_idx}")
        os.makedirs(feature_dir, exist_ok=True)

        # Save images
        for rank, (activation_val, img_path, dataset) in enumerate(top_activations):
            img_name = f"{img_path.replace('/', '_')}"
            try:
                shutil.copy(img_path, os.path.join(feature_dir, img_name))
            except Exception as e:
                print(f"Error copying files for feature {feature_idx}: {e}")

        # Save metadata (image-only)
        metadata_path = os.path.join(feature_dir, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Feature {feature_idx}\n")
            f.write(f"Total activations: {len(activations_list)}\n")
            f.write(f"Top {len(top_activations)} activations saved\n\n")

            f.write("Activation values:\n")
            for rank, (activation_val, img_path, dataset) in enumerate(top_activations):
                f.write(f"{rank+1}. {activation_val:.6f} - {dataset}/{os.path.basename(img_path)}\n")

    # Print statistics
    print(f"\n{'=' * 60}")
    print("Feature Extraction Statistics")
    print(f"{'=' * 60}")
    print(f"Total features with activations: {num_features_with_activations}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"\n✓ Feature extraction complete!")


# =================== Main ===================

if __name__ == '__main__':
    model_num = config['general']['model_num']

    all_folders = ["your_path_here/human_correct", "your_path_here/human_error"]

    # Configuration
    TARGET_LEVEL = config['general'].get('target_level', None)  # None = all levels, 0-4 = specific level
    ACTIVATION_THRESHOLD = config['general'].get('activation_threshold', 0.0)
    MAX_IMAGES_PER_FEATURE = config['general'].get('max_images_per_feature', 20)

    print(f"\n{'=' * 60}")
    print(f"Feature Extraction (IMAGE-ONLY)")
    print(f"{'=' * 60}")
    print(f"Model: {model_num}")
    print(f"Target level: {TARGET_LEVEL if TARGET_LEVEL is not None else 'All levels'}")
    print(f"Activation threshold: {ACTIVATION_THRESHOLD}")
    print(f"Max images per feature: {MAX_IMAGES_PER_FEATURE}")
    print(f"Datasets: {', '.join(all_folders)}")

    populate_features(
        model_num=model_num,
        all_folders=all_folders,
        target_level=TARGET_LEVEL,
        activation_threshold=ACTIVATION_THRESHOLD,
        max_images_per_feature=MAX_IMAGES_PER_FEATURE
    )
