import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)


def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam, a, u_dc):
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Train ridge regression
    W = ridge_closed_form(X_train_b, Y_train, lam)

    # Handle 1D vs 2D labels
    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    # Predictions
    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
    results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results, conf_matrix


def compute_feature_ranking_autoencoder(X, hidden_dim=512, epochs=50, batch_size=128, lr=0.001, l1_penalty=0.001, random_state=42):
    """
    Compute feature ranking using Sparse Autoencoder (unsupervised).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features).
    hidden_dim : int
        Size of hidden layer (should be < n_features for compression).
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    l1_penalty : float
        L1 regularization for sparsity.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ranked_idx : np.ndarray
        Indices of features sorted by importance (descending).
    importances : np.ndarray
        Mean absolute encoder weights per feature.
    """
    torch.manual_seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize X (optimization: helps training)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    class SparseAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.decoder = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            encoded = torch.relu(self.encoder(x))
            return torch.sigmoid(self.decoder(encoded))

    model = SparseAutoencoder(X.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(epochs):
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, inputs) + l1_penalty * torch.norm(model.encoder.weight, p=1)
            loss.backward()
            optimizer.step()

    # Importance: mean abs weights from encoder
    importances = torch.mean(torch.abs(model.encoder.weight), dim=0).cpu().numpy()
    ranked_idx = np.argsort(importances)[::-1]

    return ranked_idx, importances



def process_top_k(top_k, ranked_idx, X_train, X_val, labels_train, labels_val, lambda_value, a_value, u_dc_value, results_dir):
    selected_idx = ranked_idx[:top_k]
    X_train_selected = X_train[:, selected_idx]
    X_val_selected   = X_val[:, selected_idx]

    results, conf_matrix = ridge_regression_fast(
        X_train_selected, labels_train, X_val_selected, labels_val, 
        lambda_value, a_value, u_dc_value
    )

    # Save results
    np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt",
               results.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt",
               conf_matrix, fmt="%.5f")


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    lambda_value = 1e4
    top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
    mu = 1.0

    # Load training, validation
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/auto-encoder/2600/results"
    os.makedirs(results_dir, exist_ok=True)

    # Compute RF scores once
    ranked_idx, rf_scores = compute_feature_ranking_autoencoder(X_train, labels_train)

    # Run in parallel
    Parallel(n_jobs=16)(
        delayed(process_top_k)(
            top_k, ranked_idx, X_train, X_val, labels_train, labels_val,
            lambda_value, a_value, u_dc_value, results_dir
        )
        for top_k in top_k_values
    )