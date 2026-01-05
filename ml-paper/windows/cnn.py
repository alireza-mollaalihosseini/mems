import os
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy


# Load
X_train = np.load("/scratch/almo2783/scratch/ml-paper/windows/state-matrix/train-state-a-0.44-win-10.npz")["arr_0"]
X_val   = np.load("/scratch/almo2783/scratch/ml-paper/windows/state-matrix/val-state-a-0.44-win-10.npz")["arr_0"]
y_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
y_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

# Convert from one-hot to class indices (required by PyTorch cross entropy)
y_train = np.argmax(y_train, axis=1)
y_val   = np.argmax(y_val, axis=1)

# Flatten TEMPORARILY for scaling only
N, T, n_feat = X_train.shape # N, T, F = X_train.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, n_feat)
X_val   = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape[0], T, n_feat)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

def make_loader(ds, batch):
    return DataLoader(
        ds, batch_size=batch, shuffle=True,
        num_workers=8, pin_memory=True
    )



class CNNClassifier(pl.LightningModule):

    def __init__(self, lr, num_filters, kernel_size, fc_dim, dropout):
        super().__init__()
        self.save_hyperparameters()

        nf = num_filters
        ks = kernel_size

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, ks, padding=ks//2),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # ✅ 10 temporal windows = input channels
        self.conv1 = block(10, nf)
        self.conv2 = block(nf, nf*2)
        self.conv3 = block(nf*2, nf*4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(nf*4, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 10),
        )

        self.lr = lr


    def forward(self, x):
        # x = (batch, 10, 2400)

        x = self.conv1(x)
        x = F.max_pool1d(x, 2)

        x = self.conv2(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()               # SAFETY FIX
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)



torch.set_float32_matmul_precision("high")


def objective(trial):

    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    fc_dim = trial.suggest_categorical("fc_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = CNNClassifier(lr, num_filters, kernel_size, fc_dim, dropout)

    train_loader = make_loader(train_ds, batch_size)
    val_loader = make_loader(val_ds, batch_size)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=False),
        precision="16-mixed",
        max_epochs=20,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=False,
        benchmark=True,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_acc"].item()



study = optuna.create_study(
        direction="maximize",
        study_name="CNN_10-Windows-Optimization",
        storage="sqlite:///cnn-10win.db",
        load_if_exists=True
    )

study.optimize(objective, n_trials=600, gc_after_trial=True)