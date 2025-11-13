# Matryoshka Transcoders

A framework for training **Matryoshka Transcoders** - hierarchical sparse autoencoders that learn interpretable features for detecting physical plausibility errors in images.

## Overview

This repository provides tools to:
- Train classifiers to distinguish physically plausible from implausible images
- Train Matryoshka Transcoders with nested multi-level latent representations
- Extract and analyze interpretable features using Large Multimodal Models (LMMs)
- Use trained models for physical plausibility inference
- Compare with baseline methods (standard transcoders, sparse autoencoders)

## Key Features

✨ **Hierarchical Sparse Representations**: Nested levels (128 → 256 → 512 → 1024 → 2048 latents)
🔍 **Interpretable Features**: Each latent dimension corresponds to a semantic concept
🤖 **LMM-Enhanced Analysis**: Use vision-language models to interpret features
📊 **Physical Plausibility Detection**: Identify anatomical errors, impossible physics, and generation artifacts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/matryoshka_transcoders.git
cd matryoshka_transcoders

# Install dependencies
pip install torch torchvision transformers pillow numpy pyyaml openai
```

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- Transformers (for CLIP)
- OpenAI API key (for LMM analysis via OpenRouter)

## Repository Structure

```
matryoshka_transcoders/
├── training/
│   ├── train_classifier_model.py          # Train CLIP-based classifier
│   ├── train_matryoshka_transcoders.py    # Train Matryoshka Transcoder
│   ├── populate.py                         # Extract top-activating images per feature
│   ├── LMM_interpretation.py               # LMM-based feature interpretation
│   └── LMM_error_interpretation.py         # LMM-based error analysis
├── inference/
│   └── image_labeling_matryoshka.py        # Label new images with trained model
├── baselines/
│   ├── train_transcoder.py                 # Standard transcoder baseline
│   ├── train_sae.py                        # Sparse autoencoder baseline
│   └── train_matryoshka_sae.py             # Matryoshka SAE baseline
├── config.yaml.example                      # Configuration template
└── README.md
```

## Quick Start

### 1. Prepare Configuration

```bash
cp config.yaml.example config.yaml
# Edit config.yaml to set your CUDA devices and paths
```

### 2. Train Classifier (Optional but Recommended)

Train a CLIP-based classifier to distinguish physically plausible from implausible images:

```bash
cd training
python train_classifier_model.py
```

**Inputs:**
- Correct images: `./datasets/human_correct/`
- Error images: `./datasets/human_error/`

**Output:**
- `clip_classifier.pt` - Trained classifier weights

### 3. Train Matryoshka Transcoder

Train the main Matryoshka Transcoder model:

```bash
cd training
python train_matryoshka_transcoders.py
```

**What happens:**
- Loads CLIP model and trained classifier
- Trains encoder (h_2 → latent space) with nested sparsity
- Trains decoders (latent → h_1) for each level
- Saves models to `./models/`

**Architecture:**
- Input: h_2 (768-dim CLIP embeddings)
- Output: h_1 (256-dim hidden layer features)
- Latent: Hierarchical sparse representation with 5 levels
  - Level 0: 128 latents (top-16)
  - Level 1: 256 latents (top-32)
  - Level 2: 512 latents (top-64)
  - Level 3: 1024 latents (top-128)
  - Level 4: 2048 latents (top-256)

### 4. Extract Features

Extract and save top-activating images for each learned feature:

```bash
cd training
python populate.py
```

**Inputs:**
- Trained transcoder from `./models/`
- Image datasets from `./datasets/`

**Output:**
```
./transcoder_features_model{N}/
├── feature_0/
│   ├── image1.png
│   ├── image2.png
│   └── metadata.txt
├── feature_1/
└── ...
```

Each feature folder contains the top-20 images that most strongly activate that feature.

### 5. LMM-Based Feature Interpretation

Use Large Multimodal Models to understand what each feature represents:

```bash
cd training

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"
# Or edit the API key directly in LMM_interpretation.py

# Identify commonalities across feature images
python LMM_interpretation.py --method matryoshka_transcoders

# Analyze physical plausibility errors
python LMM_error_interpretation.py --method matryoshka_transcoders
```

**LMM_interpretation.py** identifies what visual concept each feature represents:
```json
{
  "feature_num": "42",
  "explanation": "Animal wildlife in natural habitats",
  "num_samples": 847
}
```

**LMM_error_interpretation.py** analyzes whether features capture physical plausibility errors:
```json
{
  "feature_num": "137",
  "commonality": "Human hands and fingers",
  "error_analysis": "Error: Incorrect number of fingers or distorted hand anatomy",
  "num_samples": 523
}
```

**Output files:**
- `{method}_feature_analysis.jsonl` - Feature commonalities
- `{method}_error_analysis.jsonl` - Physical plausibility assessments

### 6. Use Model for Inference

Label new images with the trained model:

```bash
cd inference
python image_labeling_matryoshka.py
```

**Example usage in code:**

```python
from image_labeling_matryoshka import MatryoshkaTranscoderLabeling
from PIL import Image

# Initialize model
model = MatryoshkaTranscoderLabeling(
    probe_model_path="./models/matryoshka_transcoder_best.pt",
    features_info_path="./feature_analysis.jsonl"
)

# Label an image
image = Image.open("test_image.png")
explanations = model.get_explanations(image)

# Print activated features
for feat in explanations:
    print(f"Feature {feat['feature_num']}: {feat['explanation']}")
    print(f"Activation: {feat['activation']:.2f}")
```

## Baseline Methods

Compare Matryoshka Transcoders with baseline approaches:

### Standard Transcoder

```bash
cd baselines
python train_transcoder.py
```

A simple transcoder with a single latent space (no hierarchical structure).

### Sparse Autoencoder (SAE)

```bash
cd baselines
python train_sae.py
```

A sparse autoencoder that reconstructs inputs directly (h_2 → latent → h_2).

### Matryoshka SAE

```bash
cd baselines
python train_matryoshka_sae.py
```

Hierarchical sparse autoencoder that reconstructs inputs at each level (combines Matryoshka structure with SAE approach).

## Configuration

Edit `config.yaml`:

```yaml
general:
  cuda_devices: "0"                # GPU to use
  model_num: 1                     # Model identifier
  num_epochs: 100                  # Training epochs
  batch_size: 256                  # Batch size
  learning_rate: 0.0001            # Learning rate
  activation_threshold: 0.0        # Feature activation threshold
  max_images_per_feature: 20       # Images to save per feature
  all_dataset:                     # Paths to datasets
    - "./datasets/human_correct"
    - "./datasets/human_error"
```

## Training Details

### Loss Function

```
Total Loss = Σ(level_weight[i] × MSE(h_1_recon[i], h_1)) + λ × L1(latents)
```

- **Reconstruction loss**: Weighted MSE at each hierarchical level
- **Sparsity penalty**: L1 regularization on latent activations
- **Level weights**: [0.1, 0.15, 0.2, 0.25, 0.3] (sum to 1.0)

### Nested Top-k Sparsity

Each level independently selects its top-k most activated features:
- Level 0 (128 dims): top-16
- Level 1 (256 dims): top-32
- Level 2 (512 dims): top-64
- Level 3 (1024 dims): top-128
- Level 4 (2048 dims): top-256

### JumpReLU Activation

Custom activation function for sparse representations:

```python
JumpReLU(x) = ReLU(x) + β × (x > γ)
```

Default: γ = 1.0, β = 1.0

## LMM Analysis Details

### Feature Interpretation Pipeline

1. **Sample images**: Randomly select 20 images per feature
2. **Prompt LMM**: "Analyze commonalities among these images"
3. **Extract explanation**: Parse LMM response for concise feature description
4. **Save results**: Store in JSONL format

### Error Analysis Pipeline

1. **Load feature commonalities**: From previous interpretation step
2. **Sample images**: Same 20-image sampling strategy
3. **Prompt LMM with context**:
   - Feature commonality
   - Error ratio (% of error images)
   - Specific physical error categories to check
4. **Extract assessment**: "Yes" or "No common errors"
5. **Save detailed analysis**: Include error descriptions

### API Setup

Set your OpenRouter API key in the LMM scripts:

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY_HERE",  # Replace this
)
```

Or use environment variable:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

The scripts use `anthropic/claude-3-haiku` for efficient multimodal analysis.

## Results

### LMM-Based Relevance Scores

Computed from LMM error assessments:
- **Matryoshka Transcoders**: 34.7% relevant (710/2046 features)
- **Sparse Autoencoders**: 40.4% relevant (235/581 features)

Compare with threshold-based relevance (≥95% error ratio): ~16%

The LMM method is more comprehensive because it:
- Understands semantic meaning beyond pixel patterns
- Identifies physical errors even with <95% error ratio
- Filters out non-physical patterns (e.g., style, color)

### Common Physical Plausibility Errors Detected

- Incorrect number of fingers or toes
- Distorted facial features or anatomy
- Impossible physical configurations
- Objects floating without support
- Incorrect shadows or lighting
- Extra limbs or body parts
- Unnatural poses or perspectives

## Troubleshooting

**Out of memory during training:**
- Reduce batch size in config.yaml
- Use gradient accumulation
- Train on smaller dataset

**LMM API errors:**
- Verify your OpenRouter API key
- Check rate limits (add delays if needed)
- Ensure internet connection

**Features not interpretable:**
- Increase training epochs
- Adjust sparsity weight (λ)
- Try different top-k values
- Collect more diverse training data

**Model not detecting errors:**
- Verify classifier is trained properly
- Check that error dataset contains clear physical violations
- Ensure balanced training data

## Citation

If you use this code in your research, please cite:

```bibtex
@article{matryoshka_transcoders,
  title={Matryoshka Transcoders for Physical Plausibility Assessment},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- CLIP model from OpenAI
- OpenRouter for LLM API access
- PyTorch team for the framework
