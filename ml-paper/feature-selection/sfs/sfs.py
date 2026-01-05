import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import BaseEstimator, ClassifierMixin


class RidgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lam=1e4):
        self.lam = lam

    def get_params(self, deep=True):
        return {'lam': self.lam}

    def fit(self, X, y):
        if y.ndim == 1:
            classes = np.unique(y)
            n_classes = len(classes)
            Y = np.eye(n_classes)[y.astype(int)]
        else:
            Y = y
            classes = np.arange(Y.shape[1])
        
        X_b = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
        n_features = X_b.shape[1]
        I = np.eye(n_features, dtype=X.dtype)
        self.W_ = np.linalg.solve(X_b.T @ X_b + self.lam * I, X_b.T @ Y)
        self.classes_ = classes
        return self

    def predict(self, X):
        X_b = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
        y_pred = X_b @ self.W_
        return self.classes_[np.argmax(y_pred, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1) if y.ndim > 1 else y
        return accuracy_score(y_true, y_pred)


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


def process_top_k(top_k, X_train, X_val, labels_train, labels_val, lambda_value, a_value, u_dc_value, results_dir, direction):
    n_features = X_train.shape[1]
    if top_k >= n_features:
        selected_features_idx = np.arange(n_features)
    else:
        estimator = RidgeClassifier(lam=lambda_value)
        sfs = SequentialFeatureSelector(
            estimator, 
            n_features_to_select=top_k, 
            direction=direction, 
            cv=3, 
            n_jobs=-1
        )
        sfs.fit(X_train, labels_train)
        selected_features_idx = sfs.get_support(indices=True)
    
    print(f'{direction} top_k={top_k}, Length of selected indices = {len(selected_features_idx)}')
    X_train_selected = X_train[:, selected_features_idx]
    X_val_selected   = X_val[:, selected_features_idx]

    results, conf_matrix = ridge_regression_fast(
        X_train_selected, labels_train, X_val_selected, labels_val, lambda_value, a_value, u_dc_value
    )

    # Save results
    np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-{direction}-topk-{top_k}-lambda-{lambda_value}.txt",
               results.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-{direction}-topk-{top_k}-lambda-{lambda_value}.txt",
               conf_matrix, fmt="%.5f")


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    lambda_value = 1e4
    top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000]
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

    # Results base dir
    base_results_dir = f"/scratch/almo2783/scratch/ml-paper/feature-selection/sfs/results"
    directions = ['forward', 'backward']

    for direction in directions:
        results_dir = f"{base_results_dir}/results-{direction}"
        os.makedirs(results_dir, exist_ok=True)

        # Run in parallel
        Parallel(n_jobs=64, verbose=10, backend='threading')(  # adjust n_jobs depending on your cluster
            delayed(process_top_k)(
                top_k, X_train, X_val, labels_train, labels_val,
                lambda_value, a_value, u_dc_value, results_dir, direction
            )
            for top_k in top_k_values
        )