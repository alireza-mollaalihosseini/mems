import os
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def process_file(fname):
    data, _ = sf.read(fname)
    data = data.astype(np.float32)

    # DC removal
    data -= np.mean(data)

    fft_vals = np.fft.rfft(data)
    return np.log10(np.abs(fft_vals)+1e-16).astype(np.float32)


def build_state_matrices(train_file_list_path,
                         val_file_list_path,
                         test_file_list_path):

    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames   = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames  = np.loadtxt(test_file_list_path, dtype=str)

    all_filenames = np.concatenate([
        train_filenames,
        val_filenames,
        test_filenames
    ])

    results = Parallel(
        n_jobs=64,
        backend="threading",
        verbose=1
    )(
        delayed(process_file)(fname)
        for fname in all_filenames
    )

    state_matrix = np.vstack(results)
    return state_matrix


class AudioVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(X,
              latent_dim=32,
              epochs=80,
              batch_size=256,
            #   lr=1e-3,
              lr = 5e-4,
              beta = 0.1):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

    model = AudioVAE(
        input_dim=X.shape[1],
        latent_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(x_hat, batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Total: {loss_sum/len(loader):.6e} | "
            f"Recon: {recon_sum/len(loader):.6e} | "
            f"KL: {kl_sum/len(loader):.6e}"
        )

    return model


def extract_latent_mu(model, X):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        mu, _ = model.encode(X_tensor)

    return mu.cpu().numpy()


def cluster_latent(Z, n_clusters=10):
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=42
    )
    cluster_ids = kmeans.fit_predict(Z)
    return cluster_ids
    

def evaluate_unsupervised(y_true, cluster_ids):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    ari = adjusted_rand_score(y_true, cluster_ids)
    nmi = normalized_mutual_info_score(y_true, cluster_ids)

    return ari, nmi


if __name__ == "__main__":

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    train_files = "/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv"
    val_files   = "/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv"
    test_files  = "/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv"

    # OPTIONAL: labels only for evaluation
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels = np.concatenate([labels_train, labels_val, labels_test])

    # --------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------
    print("Building state matrix...")
    state_matrix = build_state_matrices(
        train_files,
        val_files,
        test_files
    )

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------
    scaler = StandardScaler()
    X_std = scaler.fit_transform(state_matrix)

    X_std = np.nan_to_num(
        X_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------
    # Train VAE (UNSUPERVISED)
    # --------------------------------------------------
    vae = train_vae(
        X_std,
        latent_dim=32,
        epochs=80,
        batch_size=256,
        # lr=1e-3,
        lr=5e-4,
        beta=0.1
    )

    # --------------------------------------------------
    # Latent Space (μ)
    # --------------------------------------------------
    Z_mu = extract_latent_mu(vae, X_std)
    np.save("latent_mu.npy", Z_mu)

    # --------------------------------------------------
    # Clustering
    # --------------------------------------------------
    cluster_ids = cluster_latent(Z_mu, n_clusters=10)
    np.save("cluster_ids.npy", cluster_ids)

    # --------------------------------------------------
    # Optional Evaluation
    # --------------------------------------------------
    ari, nmi = evaluate_unsupervised(labels, cluster_ids)
    print(f"ARI = {ari:.4f}")
    print(f"NMI = {nmi:.4f}")

    print("\n✅ Unsupervised VAE audio scene discovery finished successfully.")



