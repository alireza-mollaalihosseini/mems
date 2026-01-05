import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import trange

# -----------------------
# Force CPU only
# -----------------------
torch.set_num_threads(8)
device = torch.device("cpu")

# -----------------------
# Load data
# -----------------------
X_train = np.load("/scratch/almo2783/scratch/ml-paper/windows/state-matrix/train-state-a-0.44-win-10.npz")["arr_0"]
X_val   = np.load("/scratch/almo2783/scratch/ml-paper/windows/state-matrix/val-state-a-0.44-win-10.npz")["arr_0"]
y_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
y_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

# One-hot -> class labels
y_train = np.argmax(y_train, axis=1)
y_val   = np.argmax(y_val, axis=1)

# -----------------------
# Scale per-sample
# -----------------------
N, T, n_feat = X_train.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, n_feat)
X_val   = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape[0], T, n_feat)

# Torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)

# Datasets + loaders
batch_size = 64

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=False
)

# -----------------------
# Single CNN Model
# -----------------------
class CNNClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        num_filters = 256
        kernel_size = 5
        dropout = 0.2
        fc_dim = 256

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # 10 temporal windows are channels
        self.conv1 = block(10, num_filters)
        self.conv2 = block(num_filters, num_filters*2)
        self.conv3 = block(num_filters*2, num_filters*4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_filters*4, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x, 2)

        x = self.conv2(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)

# -----------------------
# Setup training
# -----------------------
model = CNNClassifier().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

epochs = 20

# -----------------------
# Train loop (CPU)
# -----------------------
for epoch in trange(epochs, desc="Training"):

    # ---- Training ----
    model.train()
    train_acc = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(1) == y).float().mean().item()
        train_acc.append(acc)

    # ---- Validation ----
    model.eval()
    val_acc = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            acc = (logits.argmax(1) == y).float().mean().item()
            val_acc.append(acc)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train acc: {np.mean(train_acc):.4f} | "
        f"Val acc:   {np.mean(val_acc):.4f}"
    )

# -----------------------
# Final accuracy
# -----------------------
print("\nFinal validation accuracy:", np.mean(val_acc))
