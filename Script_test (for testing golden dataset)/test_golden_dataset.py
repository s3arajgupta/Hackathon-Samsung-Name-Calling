# -*- coding: utf-8 -*-
"""Test_golden_dataset.py

A Python program for evaluating the golden dataset with the trained
Siamese Twin CNN Name-Calling model.

Developed by Team Congruent (Samsung PRISM Program):
    - Swaraj Gupta (swaraj.gupta217@gmail.com)
    - Shiv Jaiswal (shivjaiswal2000@gmail.com)
    - Mayank Goyal (goyalmayank522@gmail.com)

Features:
    - Zero-ASR Acoustic Feature Extraction (Mel-Spectrograms, 16kHz, 1.0s)
    - Siamese Twin CNN Inference via Euclidean Distance (< 0.5 = Match)
    - Compatible with both Local execution and Google Colab environments
    - Command-line argument support with automatic local fallback discovery
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Suppress verbose TensorFlow warnings where possible
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    import librosa
except ImportError:
    librosa = None

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None

# Optional Google Colab drive mount (graceful fallback when run locally)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, Exception):
    pass


def scale_minmax(X, min_val=0.0, max_val=1.0):
    """Normalize spectrogram matrix between min_val and max_val."""
    x_std = (X - X.min()) / (X.max() - X.min() + 1e-8)
    return x_std * (max_val - min_val) + min_val


def extract_mel_feature(audio_path, target_sr=16000, target_duration=1):
    """Extract and normalize a 128-band Mel-spectrogram from a 1-second WAV file."""
    y, sr = librosa.load(audio_path, mono=True, duration=target_duration, sr=target_sr)
    target_samples = target_sr * target_duration
    if len(y) < target_samples:
        pad_width = target_samples - len(y)
        y = np.pad(y, (0, pad_width), 'constant')
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    img = scale_minmax(mel, 0, 255).astype(np.uint8)
    img_array = np.expand_dims(img, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


def find_default_path(candidates, description="file"):
    """Search candidate paths and return the first existing one."""
    for c in candidates:
        if c and os.path.exists(c):
            return str(c)
    # Return first non-None candidate as fallback
    for c in candidates:
        if c:
            return str(c)
    return ""


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    default_model = find_default_path([
        repo_root / "model",
        script_dir / "model",
        Path("model"),
        "/content/drive/MyDrive/prism_model/model_congruent",
        "/content/model"
    ], "model directory")

    default_input = find_default_path([
        script_dir / "textinput.txt",
        Path("textinput.txt"),
        "/content/textinput.txt",
        "/content/drive/MyDrive/prism_dataset/textinput.txt"
    ], "input.txt")

    default_output = str(script_dir / "txtoutput.txt")

    parser = argparse.ArgumentParser(
        description="Evaluate Golden Dataset using the trained Siamese Twin Name-Calling CNN"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Path to trained SavedModel directory or .h5 file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=default_input,
        help="Path to input text file formatted as: <serial_no> <ref_wav_path> <query_wav_path>"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Path to output text file for predictions (<serial_no> \\t <0_or_1>)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Euclidean distance threshold (default: 0.5; distance < 0.5 indicates match)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Samsung PRISM - Name Calling Golden Dataset Evaluator (Team Congruent)")
    print("=" * 70)
    print(f"Model Path       : {args.model}")
    print(f"Input File       : {args.input}")
    print(f"Output File      : {args.output}")
    print(f"Match Threshold  : {args.threshold} (distance < {args.threshold} => Match)")
    print("=" * 70)

    if librosa is None or keras is None:
        missing = []
        if librosa is None:
            missing.append("librosa")
        if keras is None:
            missing.append("tensorflow")
        print(f"Error: Missing required package(s): {', '.join(missing)}.")
        print(f"Please install them via: pip install {' '.join(missing)}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at '{args.model}'.")
        print("Please specify a valid path using --model <path_to_model>.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'.")
        print("Please specify a valid path using --input <path_to_input.txt>.")
        sys.exit(1)

    print("Loading Siamese CNN model...")
    model = keras.models.load_model(args.model, compile=False)
    print("Model successfully loaded!\n")

    results = []
    with open(args.input, "r", encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    print(f"Found {len(lines)} test pair(s) in {args.input}. Starting inference...\n")

    for line in lines:
        words = line.split()
        if len(words) < 3:
            continue

        serial_no = words[0]
        reference_path = words[1]
        query_path = words[2]

        # Check if paths exist
        if not os.path.exists(reference_path):
            print(f"[{serial_no}] Warning: Reference audio not found at '{reference_path}'. Skipping.")
            continue
        if not os.path.exists(query_path):
            print(f"[{serial_no}] Warning: Query audio not found at '{query_path}'. Skipping.")
            continue

        try:
            ref_img_array = extract_mel_feature(reference_path)
            que_img_array = extract_mel_feature(query_path)

            # Predict Euclidean distance between the twin representations
            distance = model.predict([ref_img_array, que_img_array], verbose=0)
            distance_val = float(distance.ravel()[0])
            is_match = 1 if distance_val < args.threshold else 0

            print(f"[{serial_no}] Distance: {distance_val:.4f} -> Prediction: {is_match} ({'MATCH' if is_match else 'DIFFERENT'})")
            results.append((serial_no, is_match))
        except Exception as e:
            print(f"[{serial_no}] Error during feature extraction/inference: {e}")

    # Write output file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f_out:
        for serial_no, pred in results:
            f_out.write(f"{serial_no}\t{pred}\n")

    print(f"\nEvaluated {len(results)} pair(s). Predictions saved to '{args.output}'.")


if __name__ == "__main__":
    main()
