import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Ridge prediction
# ------------------------------------------------------------
def ridge_predict(X, W):
    """
    X : (n_samples, n_features)
    W : (n_features + 1, n_classes)
    """
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])  # add bias column
    scores = Xb @ W
    return np.argmax(scores, axis=1)


def remove_band_from_weights(W, start, end):
    """
    Removes rows [start:end] from the W matrix (weights), 
    but preserves the last row (bias).
    """
    W_no_bias = W[:-1]      # all weights except bias
    W_bias = W[-1:]         # bias row

    # remove feature band
    W_reduced = np.delete(W_no_bias, np.s_[start:end], axis=0)

    # add bias row back
    return np.vstack([W_reduced, W_bias])



X      = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
X_val  = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
labels_train = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_train.npy")
labels_val   = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_val.npy")
weights = np.load("/scratch/almo2783/scratch/ml-paper/new-task/weights/weights-a-0.44-lambda-0.001.npz")['arr_0']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_val_std = scaler.transform(X_val)

X_val = X_val_std
y_val = labels_val
W = weights

y_true_int = np.argmax(y_val, axis=1)
baseline_acc = accuracy_score(y_true_int, ridge_predict(X_val, W))

print(baseline_acc)


# import sys
# import time
# import os
# import numpy as np
# from joblib import Parallel, delayed
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler

# def ridge_closed_form(X_train, Y_train, lam):
#     n_features = X_train.shape[1]
#     I = np.eye(n_features, dtype=X_train.dtype)
#     return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)


# def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam, a, u_dc):
#     # Add bias term
#     X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
#     X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

#     # Train ridge regression
#     W = ridge_closed_form(X_train_b, Y_train, lam)

#     # Handle 1D vs 2D labels
#     y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
#     y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

#     # Predictions
#     y_train_pred = X_train_b @ W
#     y_train_hats = np.argmax(y_train_pred, axis=1)
#     train_accuracy = np.mean(y_train_hats == y_train_true)

#     y_eval_pred = X_eval_b @ W
#     y_eval_hats = np.argmax(y_eval_pred, axis=1)

#     accuracy  = accuracy_score(y_eval_true, y_eval_hats)
#     precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
#     recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
#     f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

#     conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
#     results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
#     return results, conf_matrix



# lambda_value = 1e-3
# a_value = 0.44
# u_dc_value = 0.4

# X_train = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
# X_val   = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
# labels_train = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_train.npy")
# labels_val   = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_val.npy")

# # Standardize
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val   = scaler.transform(X_val)

# results, conf_matrix = ridge_regression_fast(
#         X_train, labels_train, X_val, labels_val, lambda_value, a_value, u_dc_value
#     )

# print(results)