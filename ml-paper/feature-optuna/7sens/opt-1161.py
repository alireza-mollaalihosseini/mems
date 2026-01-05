import sys
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
import pandas as pd
import json
from joblib import Parallel, delayed


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
    if Y_train.ndim == 1:
        y_train_true = Y_train
    else:
        y_train_true = np.argmax(Y_train, axis=1)

    if Y_eval.ndim == 1:
        y_eval_true = Y_eval
    else:
        y_eval_true = np.argmax(Y_eval, axis=1)

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


def top_features(W, top_k=10):
    # Exclude bias (last row of W)
    W_no_bias = W[:-1, :]

    # Dict to store max importance per feature across all classes
    feature_importance = {}

    # Loop over each class (columns of W)
    for c in range(W_no_bias.shape[1]):
        importance = np.abs(W_no_bias[:, c])
        top_idx = np.argsort(importance)[::-1][:top_k]

        # Save the *maximum importance* seen across classes
        for idx in top_idx:
            if idx not in feature_importance:
                feature_importance[idx] = importance[idx]
            else:
                feature_importance[idx] = max(feature_importance[idx], importance[idx])

    # Sort collected unique indices by importance (descending)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Pick exactly top_k
    top_idx_final = np.array([idx for idx, _ in sorted_features[:top_k]])
    
    return top_idx_final


def objective(trial, X_train, weights, X_val, labels_train, labels_val, a_value, u_dc_value):
    # Suggest number of features to select
    n_features = trial.suggest_int('n_features', 10, 5000)
    
    # Suggest regularization parameter
    lam = trial.suggest_float('lambda', 1e-5, 1e6, log=True)
    
    # Feature selection using top_features on weights
    selected_features = top_features(weights, top_k=min(n_features, weights.shape[0] - 1))  # -1 for bias exclusion
    X_train_sel = X_train[:, selected_features]
    X_val_sel = X_val[:, selected_features]
    
    results, _ = ridge_regression_fast(
        X_train_sel, labels_train, X_val_sel, labels_val, lam, a_value, u_dc_value
    )
    
    # Set additional metrics as user attributes for later export
    trial.set_user_attr('train_accuracy', float(results[3]))
    trial.set_user_attr('precision', float(results[5]))
    trial.set_user_attr('recall', float(results[6]))
    trial.set_user_attr('f1', float(results[7]))
    
    # Objective: maximize validation accuracy
    return float(results[4])


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.1
    mu = 1.0

    # Load training, validation, and testing state matrices and label matrices
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_train-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_val-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Load weights
    weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Ensure results directory exists
    results_dir = f"/scratch/almo2783/scratch/ml-paper/feature-optuna/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create or resume Optuna study with SQLite storage
    study = optuna.create_study(
        study_name="ridge_feature_selection_weights_1161",
        direction="maximize",
        storage=f"sqlite:///{results_dir}/optuna-weights-1161.db",
        load_if_exists=True
    )
    
    # Optimize (adjust n_trials as needed)
    study.optimize(
        lambda trial: objective(trial, X_train, weights, X_val, labels_train, labels_val, a_value, u_dc_value),
        n_trials=1000
    )
    
    # # Export trials to CSV
    # metrics = []
    # for trial in study.trials:
    #     if trial.state == optuna.trial.TrialState.COMPLETE:
    #         row = {**trial.params, 'accuracy': trial.value}
    #         row.update(trial.user_attrs)
    #         metrics.append(row)
    
    # df = pd.DataFrame(metrics)
    # df.to_csv(f"{results_dir}/optuna_results_weights.csv", index=False)
    
    # # Save best parameters to JSON
    # best_params = study.best_params
    
    # # To save selected features, load weights and select with best n_features
    # selected_features = top_features(weights, top_k=min(best_params['n_features'], weights.shape[0] - 1)).tolist()
    
    # best_dict = {
    #     'n_features': best_params['n_features'],
    #     'lambda': best_params['lambda'],
    #     'selected_features': selected_features
    # }
    # with open(f"{results_dir}/best_params_weights.json", "w") as f:
    #     json.dump(best_dict, f, indent=4)
    
    # Run best model and save results/conf_matrix
    X_train_best = X_train[:, selected_features]
    X_val_best = X_val[:, selected_features]
    
    best_lam = best_params['lambda']
    best_results, best_conf_matrix = ridge_regression_fast(
        X_train_best, labels_train, X_val_best, labels_val,
        best_lam, a_value, u_dc_value
    )
    
    np.savetxt(f"{results_dir}/best_results_weights_1161.txt", 
               best_results.reshape(1, -1), fmt="%.5f")
    
    np.savetxt(f"{results_dir}/best_conf_matrix_weights_1161.txt", 
               best_conf_matrix, fmt="%.5f")
    
    print(f"Best accuracy: {study.best_value:.5f}")
    print(f"Best params: {best_params}")