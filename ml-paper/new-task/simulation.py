import os
import numpy as np
import pandas as pd

# ====================================================
# --- Configuration ---
# ====================================================
base_dir = "/scratch/almo2783/scratch/ml-paper/new-task"
dataset_dir = os.path.join(base_dir, "dataset")
filepaths_dir = os.path.join(base_dir, "filepaths")

os.makedirs(filepaths_dir, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)

# Task parameters
num_classes = 10
frequencies = np.linspace(20, 2000, num_classes)  # e.g. 20Hz–2000Hz
sampling_rate = 1_000_000  # 1 MHz
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Dataset sizes
n_train_total = 8408
n_val_total = 2102
n_test_total = 400

# Distribute samples per class as evenly as possible
def distribute_samples(total, classes):
    base = total // classes
    remainder = total % classes
    counts = [base + (1 if i < remainder else 0) for i in range(classes)]
    return counts

train_counts = distribute_samples(n_train_total, num_classes)
val_counts = distribute_samples(n_val_total, num_classes)
test_counts = distribute_samples(n_test_total, num_classes)

# ====================================================
# --- Sine-wave generator ---
# ====================================================
def generate_sine(frequency, phase):
    return np.sin(2 * np.pi * frequency * t + phase).astype(np.float32)

# ====================================================
# --- Dataset generation ---
# ====================================================
def generate_split(split_name, counts_per_class):
    split_dir = os.path.join(dataset_dir, split_name)
    file_list = []
    labels = []

    for class_idx, (freq, n_samples) in enumerate(zip(frequencies, counts_per_class)):
        for sample_idx in range(n_samples):
            phase = np.random.uniform(0, 2 * np.pi)
            signal = generate_sine(freq, phase)
            filename = f"class_{class_idx}_{sample_idx}.npy"
            filepath = os.path.join(split_dir, filename)
            np.save(filepath, signal)
            file_list.append(filepath)
            labels.append(class_idx)

    # Save filepaths
    csv_path = os.path.join(filepaths_dir, f"{split_name}-filenames.csv")
    pd.DataFrame(file_list, columns=["filepath"]).to_csv(csv_path, index=False)

    # One-hot encode labels
    label_matrix = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, lbl in enumerate(labels):
        label_matrix[i, lbl] = 1.0

    np.save(os.path.join(filepaths_dir, f"label_matrix_{split_name}.npy"), label_matrix)
    print(f"✅ {split_name}: {len(labels)} samples saved")

# ====================================================
# --- Run all splits ---
# ====================================================
generate_split("train", train_counts)
generate_split("val", val_counts)
generate_split("test", test_counts)

print("\n✅ Dataset generation complete.")
print(f"Files saved in:\n{dataset_dir}\n{filepaths_dir}")
