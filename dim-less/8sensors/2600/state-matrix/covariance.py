import numpy as np
import matplotlib.pyplot as plt

# Load data
train_path = '/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz'
train_X = np.load(train_path)['arr_0']

# Center the data (important!)
train_X_centered = train_X - np.mean(train_X, axis=0)

# Compute covariance matrix (features × features)
# This will be (24001 x 24001) — heavy in memory (~4.6 GB if float64)
# You can optionally compute a correlation matrix instead for scale invariance
cov_matrix = np.cov(train_X_centered, rowvar=False)

# Compute eigenvalues (no need for eigenvectors for now)
eigenvalues = np.linalg.eigvalsh(cov_matrix)  # more stable for symmetric matrices

# Sort in descending order
eigenvalues = np.sort(eigenvalues)[::-1]

# Compute cumulative explained variance
explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# Plot results
plt.figure(figsize=(8,5))
plt.semilogy(eigenvalues, linewidth=2)
plt.title("Eigenvalue Spectrum of Frequency Covariance Matrix", fontweight='bold', fontsize=16)
plt.xlabel("Component Index", fontweight='bold', fontsize=14)
plt.ylabel("Eigenvalue (log scale)", fontweight='bold', fontsize=14)
plt.grid(True)
plt.savefig('eigenvalue-spec.png', dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(explained_variance_ratio, linewidth=2)
plt.title("Cumulative Explained Variance (Covariance Eigenvalues)", fontweight='bold', fontsize=16)
plt.xlabel("Number of Components", fontweight='bold', fontsize=14)
plt.ylabel("Cumulative Variance Ratio", fontweight='bold', fontsize=14)
plt.grid(True)
plt.savefig('cumulative-var.png', dpi=300)
plt.show()

# Find number of components to reach 95% variance
n95 = np.argmax(explained_variance_ratio >= 0.95) + 1
print(f"Number of independent (significant) frequency components explaining 95% variance: {n95}")
