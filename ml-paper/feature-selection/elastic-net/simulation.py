import numpy as np
from sklearn.linear_model import ElasticNet, RidgeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier  # If needed for multi-class
from sklearn.base import clone
from itertools import product

a_value = 0.44
u_dc_value = 0.4
mu = 1.0

# Load training, validation
X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Elastic Net for embedded selection (treat as multi-output regression for multi-class)
# Grid for alpha (penalty) and l1_ratio (α: 0=Ridge, 1=Lasso)
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5],  # Tune based on your lambda~1e4 scale
    'l1_ratio': [0.1, 0.5, 0.9]  # Bias toward L2 for stability
}
enet = ElasticNet(max_iter=10000, random_state=42, tol=1e-4)  # Increase iter if convergence issues
grid_search = GridSearchCV(enet, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, labels_train)  # Fit on full data

# Best model
best_enet = grid_search.best_estimator_
coefs = np.abs(best_enet.coef_)  #
mean_abs_coef = np.mean(coefs, axis=0)  # Average over classes
selected_features = mean_abs_coef > 1e-5  # Stricter threshold; tune if needed
n_selected = np.sum(selected_features)
print(f"Selected {n_selected} features via Elastic Net (α={grid_search.best_params_['l1_ratio']}, lambda={grid_search.best_params_['alpha']})")

# save selected features
np.save('selected_features.npy', selected_features)

# Subset data
X_train_selected = X_train_scaled[:, selected_features]
X_val_selected = X_val_scaled[:, selected_features]

# Retrain Ridge on selected features (your classifier)
ridge = RidgeClassifier(alpha=1e4)  # Your lam
ridge.fit(X_train_selected, labels_train)

# Evaluate (as in your ridge_regression_fast)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = ridge.predict(X_val_selected)
acc = accuracy_score(labels_val, y_pred)
prec = precision_score(labels_val, y_pred, average='macro', zero_division=0)
rec = recall_score(labels_val, y_pred, average='macro', zero_division=0)
f1 = f1_score(labels_val, y_pred, average='macro', zero_division=0)
print(f"Elastic Net + Ridge: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# For multiple k (like your top_k): Post-select top-k from enet.coef_ magnitudes
# coef_mags = np.abs(best_enet.coef_).mean(axis=0)  # Average over classes if multi-output
# top_k_indices = np.argsort(coef_mags)[-k:]  # For each k