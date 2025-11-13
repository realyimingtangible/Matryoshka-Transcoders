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

def load_feature_commonalities(jsonl_file):
    """Load feature commonalities from JSONL file."""
    commonalities = {}
    if not os.path.exists(jsonl_file):
        print(f"Warning: Commonality file not found: {jsonl_file}")
        return commonalities

    with open(jsonl_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            feature_num = record.get('feature_num')
            explanation = record.get('explanation')
            if feature_num and explanation:
                commonalities[feature_num] = explanation

    print(f"Loaded {len(commonalities)} feature commonalities from {jsonl_file}")
    return commonalities

def extract_error_analysis(text):
    """Extract the error analysis description from the model response"""
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

def generate_error_analysis(samples, commonality):
    """Use Claude to analyze physical plausibility errors in sampled features."""
    # Create messages array with images
    messages = [{"role": "system", "content": "You are an expert in analyzing visual content for physical plausibility errors and anatomical accuracy."}]

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

    # Add the final analysis request with commonality context
    prompt = f"""Based on the commonality of these images: '{commonality}'

Analyze whether these images share common physical plausibility errors or anatomical inaccuracies. Look for issues such as:
- Errorful number of fingers or toes
- Misgenerated anatomical structures (e.g., extra limbs, distorted body proportions)
- Impossible physical configurations or perspectives
- Incorrect object physics or gravity
- Unnatural poses or movements
- Distorted facial features
- Other anatomical or physical errors

If there is a common error pattern across these images, identify and describe it concisely in one phrase.
If there are NO common physical plausibility errors, respond with '[No common errors]'.

Format your answer as: '[Error: Description of the common error]' or '[No common errors]'
Start your answer with '[Error:' or '[No common errors]'. End with ']'"""

    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        completion = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=messages
        )

        if completion.choices and len(completion.choices) > 0 and completion.choices[0].message:
            return extract_error_analysis(completion.choices[0].message.content)
        else:
            print(f"Error in API response: {completion}")
            return None
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def feature_error_analysis(base_path, commonalities_file, output_file):
    """
    Analyze features for physical plausibility errors.

    Args:
        base_path: Path to the directory containing feature_X folders
        commonalities_file: Path to JSONL file containing feature commonalities
        output_file: Path to output JSONL file for results
    """
    # Parameters
    SAMPLES_PER_ITERATION = 20  # Number of samples to take per iteration
    num_min_samples = 3

    # Load feature commonalities
    commonalities = load_feature_commonalities(commonalities_file)

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

        # Skip if we don't have commonality for this feature
        if feature_num not in commonalities:
            print(f"Skipping Feature {feature_num} - no commonality found")
            continue

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

        # Generate error analysis using the LLM
        commonality = commonalities[feature_num]
        print(f"Processing Feature {feature_num} ({len(png_files)} samples)...", flush=True)
        print(f"  Commonality: {commonality}", flush=True)
        error_analysis = generate_error_analysis(samples, commonality)

        # Print the result immediately
        print(f"  -> Error Analysis: {error_analysis}\n", flush=True)

        record = {
            "feature_num": feature_num,
            "commonality": commonality,
            "error_analysis": error_analysis,
            "num_samples": len(png_files)
        }

        # Save to output file
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"\nError analysis completed. Results saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze features for physical plausibility errors using LLM')
    parser.add_argument('--method', type=str,
                        choices=['matryoshka_transcoders', 'transcoders', 'sparse_autoencoders', 'matryoshka_sae'],
                        default='transcoders',
                        help='Method to analyze (default: transcoders)')
    parser.add_argument('--base_dir', type=str,
                        default='/data0/yiming_tangible/repos/matryoshka_transcoders/Feature_Results',
                        help='Base directory containing method folders')

    args = parser.parse_args()

    # Set paths based on selected method
    base_path = os.path.join(args.base_dir, args.method)
    commonalities_file = os.path.join(args.base_dir, f"{args.method}_feature_analysis.jsonl")
    output_file = os.path.join(args.base_dir, f"{args.method}_error_analysis.jsonl")

    print(f"Analyzing features from method: {args.method}")
    print(f"Base path: {base_path}")
    print(f"Commonalities file: {commonalities_file}")
    print(f"Output file: {output_file}\n")

    # Check if paths exist
    if not os.path.exists(base_path):
        print(f"Error: Path does not exist: {base_path}")
        exit(1)

    if not os.path.exists(commonalities_file):
        print(f"Error: Commonalities file does not exist: {commonalities_file}")
        exit(1)

    # Run feature error analysis
    feature_error_analysis(base_path, commonalities_file, output_file)
