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
X_train = np.load("/scratch/almo2783/scratch/ml-paper/time-domain/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")["arr_0"]
X_val   = np.load("/scratch/almo2783/scratch/ml-paper/time-domain/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")["arr_0"]
y_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
y_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

# Convert from one-hot to class indices (required by PyTorch cross entropy)
y_train = np.argmax(y_train, axis=1)
y_val   = np.argmax(y_val, axis=1)

# Standardize on CPU
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)

# CNN expects N, C, L → rearrange to (batch, 1, 5000)
X_train = X_train.unsqueeze(1)
X_val = X_val.unsqueeze(1)

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

        self.conv1 = nn.Conv1d(1, nf, ks, padding=ks//2)
        self.bn1 = nn.BatchNorm1d(nf)

        self.conv2 = nn.Conv1d(nf, nf*2, ks, padding=ks//2)
        self.bn2 = nn.BatchNorm1d(nf*2)

        self.conv3 = nn.Conv1d(nf*2, nf*4, ks, padding=ks//2)
        self.bn3 = nn.BatchNorm1d(nf*4)

        # Pool down to fixed size
        self.global_pool = nn.AdaptiveAvgPool1d(64)

        self.fc = nn.Sequential(
            nn.Linear(nf*4 * 64, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 10)
        )

        self.lr = lr

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)

        x = torch.flatten(x, 1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



def objective(trial):
    # Hyperparameters to search
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    fc_dim = trial.suggest_categorical("fc_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = CNNClassifier(lr, num_filters, kernel_size, fc_dim, dropout)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        logger=False,
        enable_checkpointing=False,
    )

    train_loader = make_loader(train_ds, batch_size)
    val_loader = make_loader(val_ds, batch_size)

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_acc"].item()



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

print("Best Params:", study.best_params)


best = study.best_params

model = CNNClassifier(
    lr=best["lr"],
    num_filters=best["num_filters"],
    kernel_size=best["kernel_size"],
    fc_dim=best["fc_dim"],
    dropout=best["dropout"],
)

train_loader = make_loader(train_ds, best["batch_size"])
val_loader = make_loader(val_ds, best["batch_size"])

logger = TensorBoardLogger("logs", name="final_model")

trainer = pl.Trainer(
    max_epochs=60,
    accelerator="gpu",
    devices="auto",
    strategy=DDPStrategy(find_unused_parameters=False),
    precision=16,
    logger=logger,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
)

trainer.fit(model, train_loader, val_loader)