import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
    return results, conf_matrix


def main(train_file_list_path, val_file_list_path, test_file_list_path, a, u_dc, mu):

    lam = 1e4

    state_train = np.load(train_file_list_path)['arr_0']
    state_val   = np.load(val_file_list_path)['arr_0']
    state_test  = np.load(test_file_list_path)['arr_0']

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Standardize using training set stats
    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(state_train)
    state_test_std  = scaler.transform(state_test)
    state_val_std   = scaler.transform(state_val)

    # Train on train set, test on test set
    results_test, cm_test = ridge_regression_fast(
        state_train_std, labels_train, state_test_std, labels_test,
        lam, a, u_dc
    )

    # Train on train set, eval on val set
    results_val, cm_val = ridge_regression_fast(
        state_train_std, labels_train, state_val_std, labels_val,
        lam, a, u_dc
    )

    # Save
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/results_test-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
               results_test.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/conf_matrix_test-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
               cm_test, fmt="%.5f")

    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/results_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
               results_val.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/conf_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
               cm_val, fmt="%.5f")



if __name__ == '__main__':
    a, u_dc, mu = (0.44, 0.4, 1.0)

    main(
        '/scratch/almo2783/scratch/ml-paper/data-augmentation/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz',
        '/scratch/almo2783/scratch/ml-paper/data-augmentation/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz',
        '/scratch/almo2783/scratch/ml-paper/data-augmentation/state-matrix/state_matrix_test-a-0.44-u_dc-0.4-mu-1.0.npz',
        a,
        u_dc,
        mu
    )