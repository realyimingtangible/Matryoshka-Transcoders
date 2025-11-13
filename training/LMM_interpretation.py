import os
import json
import random
import io
import base64
import argparse
from openai import OpenAI
from PIL import Image
import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY_HERE",
)

def extract_explanation(text):
    """Extract the commonality description from the model response"""
    if "[" in text:
        text = text.split("[")[-1]
    if "]" in text:
        text = text.split("]")[0]
    return text

def sample_files(feature_path, num_samples):
    """Randomly sample num_samples images from the feature folder."""
    # Get all PNG files in the directory
    items = [f for f in os.listdir(feature_path) if f.endswith(".png")]
    sampled_items = random.sample(items, min(num_samples, len(items)))

    samples = []
    for img_file in sampled_items:
        img_path = os.path.join(feature_path, img_file)
        if os.path.exists(img_path):
            samples.append({"image": img_path})

    return samples

def generate_explanation(samples):
    """Use Claude to analyze commonalities in sampled features."""
    # Create messages array with images
    messages = [{"role": "system", "content": "You are an expert in multimodal feature analysis."}]

    # Add each sample as a separate message with image only
    for i, sample in enumerate(samples):
        try:
            # Load and encode image
            image = Image.open(sample['image'])
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            # Resize image to reduce size
            max_size = 800
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            message_content = [
                {
                    "type": "text",
                    "text": f"Image {i+1}:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}"
                    }
                }
            ]
            messages.append({"role": "user", "content": message_content})
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Add the final analysis request
    messages.append({
        "role": "user",
        "content": "Analyze the commonalities among these images. Identify: If there exist one common feature that is possessed by all the instances. Summarize and output exactly one feature, for example: '[Commonality: Animal wildlife in natural habitats]' and '[Commonality: Strawberry-based dessert or dish]'. Only answer in general, do not analyze each image one by one, only generate one single concise phrase. Start answer with '[Commonality:'. End with ']'"
    })

    try:
        completion = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=messages
        )

        if completion.choices and len(completion.choices) > 0 and completion.choices[0].message:
            return extract_explanation(completion.choices[0].message.content)
        else:
            print(f"Error in API response: {completion}")
            return None
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def feature_analysis(base_path, output_file):
    """
    Analyze features in the transcoders directory.

    Args:
        base_path: Path to the transcoders directory containing feature_X folders
        output_file: Path to output JSONL file for results
    """
    # Parameters
    SAMPLES_PER_ITERATION = 20  # Number of samples to take per iteration
    num_min_samples = 3

    # Get all feature directories
    feature_dirs = sorted([d for d in os.listdir(base_path)
                          if os.path.isdir(os.path.join(base_path, d)) and d.startswith("feature_")],
                         key=lambda x: int(x.split("_")[1]))

    print(f"Found {len(feature_dirs)} feature directories to process")

    # Iterate through features
    for feature in feature_dirs:
        feature_path = os.path.join(base_path, feature)

        # Extract feature number
        feature_num = feature.split("_")[1]

        # Check number of samples in the feature folder
        png_files = [f for f in os.listdir(feature_path) if f.endswith(".png")]
        if len(png_files) <= num_min_samples:
            print(f"Skipping Feature {feature_num} - only {len(png_files)} sample(s)")
            continue

        # Sample images from the feature folder
        samples = sample_files(feature_path, SAMPLES_PER_ITERATION)

        if len(samples) == 0:
            print(f"Skipping Feature {feature_num} - no valid samples found")
            continue

        # Generate explanation using the LLM
        print(f"Processing Feature {feature_num} ({len(png_files)} samples)...", flush=True)
        explanation = generate_explanation(samples)

        # Print the result immediately
        print(f"  → Feature {feature_num}: {explanation}\n", flush=True)

        record = {
            "feature_num": feature_num,
            "explanation": explanation,
            "num_samples": len(png_files)
        }

        # Save to output file
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"\nAnalysis completed. Results saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze features using LLM to identify commonalities')
    parser.add_argument('--method', type=str,
                        choices=['matryoshka_transcoders', 'transcoders', 'sparse_autoencoders', 'matryoshka_sae'],
                        default='transcoders',
                        help='Method to analyze (default: transcoders)')
    parser.add_argument('--base_dir', type=str,
                        default='your_path_here',
                        help='Base directory containing method folders')

    args = parser.parse_args()

    # Set paths based on selected method
    base_path = os.path.join(args.base_dir, args.method)
    output_file = f"{args.method}_feature_analysis.jsonl"

    print(f"Analyzing features from method: {args.method}")
    print(f"Base path: {base_path}")
    print(f"Output file: {output_file}\n")

    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Error: Path does not exist: {base_path}")
        exit(1)

    # Run feature analysis
    feature_analysis(base_path, output_file)
