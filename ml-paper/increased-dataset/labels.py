import os
import numpy as np
import pandas as pd

# --- Extract scene names and build one-hot labels ---
def extract_scene_label(path):
    """Extract scene label from filename like 'airport-barcelona-0-10-7-a.wav'."""
    filename = os.path.basename(path)
    return filename.split("-")[0]  # e.g., "airport"

def one_hot_encode(label, class_list):
    """Convert label to one-hot vector."""
    vec = np.zeros(len(class_list), dtype=np.float32)
    if label in class_list:
        vec[class_list.index(label)] = 1.0
    return vec


# --- Define the 10 scene classes ---
scene_classes = [
    "airport",
    "shopping_mall",
    "metro_station",
    "street_pedestrian",
    "public_square",
    "street_traffic",
    "tram",
    "bus",
    "metro",
    "park",
]

# --- Dataset configurations ---
dataset_types = ["barcelona", "3cities", "5cities"]
splits = ["train", "test", "val"]

for dataset_type in dataset_types:
    for split in splits:
        # --- Define input CSV path ---
        csv_path = f"/scratch/almo2783/scratch/rayson/design1/{dataset_type}/{split}-filenames-{dataset_type}-rayson.csv"
        if not os.path.exists(csv_path):
            print(f"⚠️ File not found: {csv_path}")
            continue

        # --- Load dataset paths ---
        df = pd.read_csv(csv_path, header=None)
        audio_paths = df[0].tolist()

        # --- Build one-hot labels ---
        labels = []
        for path in audio_paths:
            scene = extract_scene_label(path)
            labels.append(one_hot_encode(scene, scene_classes))

        labels = np.array(labels)

        # --- Define output directory & filename ---
        save_dir = f"/scratch/almo2783/scratch/ml-paper/increased-dataset/{dataset_type}"
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, f"labels_{split}.npy")
        np.save(output_path, labels)

        print(f"✅ Saved one-hot labels: {output_path}")
        print(f"   → Shape: {labels.shape} (num_samples, num_classes={len(scene_classes)})\n")
