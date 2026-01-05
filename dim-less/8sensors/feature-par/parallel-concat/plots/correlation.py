import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Directory containing your result files ---
results_dir = "/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/lam-opt"
save_dir = "/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/plots"

# --- 2. Regex pattern to extract info from filenames ---
pattern = re.compile(
    r"results_(val|test)-sensors-([\d\-]+)-topk-(\d+)-lambda-([\deE\.\-]+)\.txt"
)

records = []

# --- 3. Walk through the directory and collect results ---
for fname in os.listdir(results_dir):
    match = pattern.match(fname)
    if not match:
        continue

    set_type, combo_str, top_k, lambda_str = match.groups()
    combo = combo_str.split("-")
    num_sensors = len(combo)
    lambda_val = float(lambda_str.replace("e", "E"))  # handle e/E notation
    top_k = int(top_k)

    # Load data
    fpath = os.path.join(results_dir, fname)
    try:
        data = np.loadtxt(fpath)
        # Suppose the file has 3 values: [train, test, val]
        # If not sure, adjust indices below
        acc_val = data[2] * 100 if len(data) > 2 else data[-1] * 100
    except Exception:
        acc_val = np.nan

    records.append({
        "set": set_type,
        "combination": combo_str,
        "num_sensors": num_sensors,
        "top_k": top_k,
        "lambda_opt": lambda_val,
        "accuracy": acc_val
    })

# --- 4. Create DataFrame ---
df = pd.DataFrame(records)
print("✅ Loaded results:", len(df))
print(df.head())

# --- 5. Save to CSV for later use ---
save_path = os.path.join(save_dir, "all_results_summary-all.csv")
df.to_csv(save_path, index=False)
print(f"Results saved to {save_path}")

# --- 6. Compute correlation ---
corr = df[["accuracy", "lambda_opt", "num_sensors", "top_k"]].corr()

# --- 7. Plot correlation heatmap ---
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation: Accuracy, λ, Sensor Count, Top-k", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

# --- 8. Plot scatter for more detailed relationship ---
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x=np.log10(df["lambda_opt"]), y="accuracy",
                hue="num_sensors", palette="viridis", s=80, alpha=0.8, edgecolor="k")
plt.xlabel("log₁₀(λ_opt)", fontsize=13, fontweight="bold")
plt.ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
plt.title("Accuracy vs Optimal λ (colored by #Sensors)", fontsize=15, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "scatter_accuracy_vs_lambda.png"), dpi=300)
plt.close()
