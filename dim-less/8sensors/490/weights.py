import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

    # Predictions
    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    y_train_true = np.argmax(Y_train, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)
    y_eval_true = np.argmax(Y_eval, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
    results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results, conf_matrix, W

if __name__ == "__main__":
    # Parse command-line arguments
    a_value = 0.04
    u_dc_value = 0.9
    lambda_value = 1e4
    mu = 1.0

    # Load training, validation, and testing state matrices and label matrices
    state_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
    # state_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
    state_test  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_test-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    # labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    # load compressed arrays
    state_train = state_train['arr_0']
    # state_val = state_val['arr_0']
    state_test = state_test['arr_0']

    # Standardize using training set stats
    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(state_train)
    state_test_std  = scaler.transform(state_test)
    # state_val_std   = scaler.transform(state_val)

    # Perform ridge regression
    # results, conf_matrix, weights = ridge_regression_fast(
    #     state_train_std, labels_train, state_val_std, labels_val,
    #     lambda_value, a_value, u_dc_value
    # )

    results, conf_matrix, weights = ridge_regression_fast(
        state_train_std, labels_train, state_test_std, labels_test,
        lambda_value, a_value, u_dc_value
    )

    # Save weights
    np.savez_compressed(f"/scratch/almo2783/scratch/dim-less/8sensors/490/weights/weights-a-{a_value}-lambda-{lambda_value}.npz", weights)