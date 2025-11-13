"""
Image labeling script using Matryoshka Transcoder relevant features.

This script loads the extracted relevant features probe and uses it to label images
with physical plausibility features. Only images are required (no text).
"""

from PIL import Image
import os
import json
import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple


# =================== Feature Probe Wrapper ===================

class FeatureProbe(nn.Module):
    """Wrapper for multiple feature probes"""
    def __init__(self, probes):
        super().__init__()
        self.probes = nn.ModuleList(probes)

    def forward(self, x):
        return torch.cat([probe(x) for probe in self.probes], dim=1)


# Trust the FeatureProbe class for safe loading
torch.serialization.add_safe_globals([FeatureProbe])


# =================== JumpReLU (needed for probe forward pass) ===================

class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.relu(x) + self.beta * (x > self.gamma).float()


# =================== Matryoshka Transcoder Labeling Model ===================

class MatryoshkaTranscoderLabeling(nn.Module):
    def __init__(self, probe_model_path, features_info_path, device=None):
        """
        Initialize the labeling model.

        Args:
            probe_model_path: Path to the relevant_features_probe.pth file
            features_info_path: Path to the relevant_features_info.jsonl file
            device: Device to run on (cuda or cpu)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model for encoding images
        print("Loading CLIP model...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=False
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=False
        )
        self.clip.eval()

        # Load feature explanations first (needed to know number of features)
        print(f"Loading feature explanations from: {features_info_path}")
        self.explanations = {}
        with open(features_info_path, 'r') as f:
            for line in f:
                feature_info = json.loads(line.strip())
                feature_idx = feature_info['feature_num']
                self.explanations[feature_idx] = {
                    'explanation': feature_info['explanation'],
                    'wrong_ratio': feature_info['wrong_ratio'],
                    'wrong_count': feature_info['wrong_count'],
                    'total_count': feature_info['total_count']
                }

        self.num_features = len(self.explanations)

        # Load feature probe model
        print(f"Loading feature probe from: {probe_model_path}")
        # Create empty FeatureProbe with correct number of probes
        probes = [nn.Linear(768, 1) for _ in range(self.num_features)]
        self.probe = FeatureProbe(probes)
        # Load the saved state dict
        state_dict = torch.load(probe_model_path, map_location=self.device, weights_only=False)
        self.probe.load_state_dict(state_dict)
        self.probe.eval()
        self.probe.to(self.device)

        # Activation threshold (can be adjusted based on your needs)
        # Typical activation values range from 0 to ~1.2
        # Threshold of 0.5
        self.threshold = 0.5

        print("=" * 80)
        print("Matryoshka Transcoder Labeling Model Initialized")
        print("=" * 80)
        print(f"Number of relevant features: {self.num_features}")
        print(f"Activation threshold: {self.threshold}")
        print(f"Device: {self.device}")
        print("=" * 80)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image using CLIP to get h_2 (768-dim embeddings).

        Args:
            image: PIL Image

        Returns:
            Image features tensor [1, 768]
        """
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_feat = self.clip.get_image_features(**inputs)

        return image_feat

    def _get_sparse_activations(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sparse feature activations for the image.

        Args:
            image: PIL Image

        Returns:
            z_sparse: Feature activations [1, num_features]
            z_mask: Boolean mask of activated features [num_features]
        """
        # Encode image to h_2 (768-dim CLIP embeddings)
        image_feat = self._encode_image(image)

        # Apply probe to get feature activations
        with torch.no_grad():
            z_sparse = self.probe(image_feat)

        # Note: Probes already include the encoder weights which have JumpReLU
        # So we just apply ReLU here for the final activation
        z_sparse = torch.relu(z_sparse)

        # Create mask for features above threshold
        z_mask = (z_sparse > self.threshold).squeeze(0)

        return z_sparse, z_mask

    def get_explanations(self, image: Image.Image) -> List[Dict]:
        """
        Get explanations for activated features in the image.

        Args:
            image: PIL Image

        Returns:
            List of dictionaries with feature index, explanation, activation value, and metadata
        """
        z_sparse, z_mask = self._get_sparse_activations(image)

        # Get indices of activated features
        activated_indices = z_mask.nonzero(as_tuple=True)[0].tolist()

        # Map back to original feature numbers
        feature_nums = list(self.explanations.keys())

        explanations = []
        for idx in activated_indices:
            if idx < len(feature_nums):
                feature_num = feature_nums[idx]
                feature_info = self.explanations[feature_num]

                explanations.append({
                    'feature_num': feature_num,
                    'activation': z_sparse[0, idx].item(),
                    'explanation': feature_info['explanation'],
                    'wrong_ratio': feature_info['wrong_ratio'],
                    'wrong_count': feature_info['wrong_count'],
                    'total_count': feature_info['total_count']
                })

        # Sort by activation value (highest first)
        explanations.sort(key=lambda x: x['activation'], reverse=True)

        return explanations

    def label_image(self, image_path: str, save_results: bool = True, output_dir: str = "./") -> List[Dict]:
        """
        Label an image with relevant physical plausibility features.

        Args:
            image_path: Path to the image file
            save_results: Whether to save results (image, labels, caption if exists)
            output_dir: Directory to save results (default: current directory)

        Returns:
            List of activated features with explanations
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get explanations
        explanations = self.get_explanations(image)

        # Print results
        print(f"\n{'=' * 80}")
        print(f"Image: {image_path}")
        print(f"{'=' * 80}")
        print(f"Number of activated features: {len(explanations)}\n")

        for i, feat in enumerate(explanations, 1):
            print(f"{i}. Feature {feat['feature_num']} (activation: {feat['activation']:.2f})")
            print(f"   {feat['explanation']}")
            print(f"   [Relevance: {feat['wrong_ratio']:.0%} wrong, {feat['wrong_count']}/{feat['total_count']}]")
            print()

        # Save results if requested
        if save_results:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get base filename without extension
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            # Save the image copy
            image_output_path = os.path.join(output_dir, f"{base_filename}.png")
            image.save(image_output_path)
            print(f"Image saved to: {image_output_path}")

            # Save labels JSON
            labels_output_path = os.path.join(output_dir, f"{base_filename}_labels.json")
            with open(labels_output_path, 'w') as f:
                json.dump({
                    'original_image_path': image_path,
                    'saved_image_path': image_output_path,
                    'num_activated_features': len(explanations),
                    'threshold': self.threshold,
                    'features': explanations
                }, f, indent=2)
            print(f"Labels saved to: {labels_output_path}")

            # Check if there's a caption/description file (.txt)
            caption_path = image_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption = f.read()

                # Save caption
                caption_output_path = os.path.join(output_dir, f"{base_filename}_caption.txt")
                with open(caption_output_path, 'w') as f:
                    f.write(caption)
                print(f"Caption saved to: {caption_output_path}")
            else:
                print("No caption file found")

        return explanations


# =================== Main ===================

if __name__ == "__main__":
    # Paths to the extracted features
    PROBE_MODEL_PATH = "your_probe_path.pth"
    FEATURES_INFO_PATH = "your_feature_path.jsonl"

    # Initialize the labeling model
    model = MatryoshkaTranscoderLabeling(
        probe_model_path=PROBE_MODEL_PATH,
        features_info_path=FEATURES_INFO_PATH
    )

    # === Example usage: Label images ===
    # Choose which image to label
    image_path = "your_image_path.png"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        exit(1)

    # Label the image
    explanations = model.label_image(image_path, save_results=True)

    print("\n" + "=" * 80)
    print("Labeling Complete!")
    print("=" * 80)
