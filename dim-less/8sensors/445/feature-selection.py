# import sys
# import time
# import numpy as np
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
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

#     # Predictions
#     y_train_pred = X_train_b @ W
#     y_train_hats = np.argmax(y_train_pred, axis=1)
#     y_train_true = np.argmax(Y_train, axis=1)
#     train_accuracy = np.mean(y_train_hats == y_train_true)

#     y_eval_pred = X_eval_b @ W
#     y_eval_hats = np.argmax(y_eval_pred, axis=1)
#     y_eval_true = np.argmax(Y_eval, axis=1)

#     accuracy  = accuracy_score(y_eval_true, y_eval_hats)
#     precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
#     recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
#     f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

#     conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
#     results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
#     return results, conf_matrix


# if __name__ == "__main__":
#     # Parse command-line arguments
#     a_value = 0.16
#     u_dc_value = 0.1
#     lambda_value = 1e4
#     alpha_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
#     # lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
#     mu = 1.0

#     # Load training, validation, and testing state matrices and label matrices
#     X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
#     X_val  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
#     # X_test  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
#     labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
#     labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
#     # labels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

#     # load compressed arrays
#     X_train = X_train['arr_0']
#     X_val = X_val['arr_0']
#     # X_test = X_test['arr_0']

#     # Standardize using training set stats
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     # X_test  = scaler.transform(X_test)
#     X_val   = scaler.transform(X_val)

#     # Load the weights
#     weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/weights/weights_val-a-0.16-lambda-10000.0.npz")
#     weights = weights['arr_0']

#     for alpha in alpha_values:

#         # List to store extracted feature indices
#         extracted_features_idx = []

#         # Iterate over the classes
#         for i in range(10):
#             # Split data into positive and negative parts
#             positive_part = [x for x in weights[i] if x > 0]
#             negative_part = [x for x in weights[i] if x < 0]

#             # Calculate median of the positive and negative parts
#             positive_median = np.median(positive_part) if len(positive_part) > 0 else 0
#             negative_median = np.median(negative_part) if len(negative_part) > 0 else 0

#             # Indices where values are greater/less than the median
#             extracted_features_positive_idx = np.where(weights[i] > alpha * positive_median)[0]
#             extracted_features_negative_idx = np.where(weights[i] < alpha * negative_median)[0]

#             # Append indices to the list
#             extracted_features_idx.extend(extracted_features_positive_idx)
#             extracted_features_idx.extend(extracted_features_negative_idx)

#         # Remove duplicate indices using np.unique
#         extracted_features_idx = np.unique(extracted_features_idx)

#         # Print the result
#         print(f"The Length of the Extracted Feature Indices with alpha = {alpha}: {int(len(extracted_features_idx))}")

#         if len(extracted_features_idx) != 0:

#             # Select columns corresponding to the extracted feature indices
#             X_train_selected = X_train[:, extracted_features_idx]
#             X_val_selected  = X_val[:, extracted_features_idx]
                
#             results, conf_matrix = ridge_regression_fast(X_train_selected, labels_train, X_val_selected, labels_val, lambda_value, a_value, u_dc_value)

#             # Save the results (scalars)
#             np.savetxt(f"/scratch/almo2783/scratch/dim-less/8sensors/445/selected_features/results/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt", results.reshape(1, -1), fmt="%.5f")
            
#             # Save confusion matrix (2D array)
#             np.savetxt(f"/scratch/almo2783/scratch/dim-less/8sensors/445/selected_features/results/conf_matrix-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt", conf_matrix, fmt="%.5f")


import sys
import time
import os
import numpy as np
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
    
    # Compute L2 norm importance
    importance = np.linalg.norm(W_no_bias, axis=1)  # shape (n_features,)
    
    # Get top-k indices
    top_idx = np.argsort(importance)[::-1][:top_k]
    
    return top_idx


if __name__ == "__main__":
    a_value = 0.16
    u_dc_value = 0.1
    lambda_value = 1e4
    # alpha_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
    #                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
    alpha_values = [10, 50, 100, 200, 300, 400, 500, 1000, 2000]
    mu = 1.0

    # Load training, validation, and testing state matrices and label matrices
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Load the weights
    weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/weights/weights-a-0.16-lambda-10000.0.npz")['arr_0']
    # # remove bias weight
    # weights = weights[:-1,:]

    # Ensure results directory exists
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/445/selected_features/results"
    os.makedirs(results_dir, exist_ok=True)

    for alpha in alpha_values:
        # abs_weights_all = np.mean(np.abs(weights), axis=1)

        # # threshold-based selection
        # threshold = alpha * np.median(abs_weights_all)
        # extracted_features_idx = np.where(abs_weights_all > threshold)[0]

        # # fallback: if too few features, keep top-50
        # if len(extracted_features_idx) < 50:
        #     topk_idx = np.argsort(abs_weights_all)[::-1][:50]
        #     extracted_features_idx = np.unique(np.concatenate([extracted_features_idx, topk_idx]))

        # print(f"Alpha={alpha:.2f} -> Extracted Features: {len(extracted_features_idx)}")

        extracted_features_idx = top_features(weights, top_k=alpha)

        if len(extracted_features_idx) > 0:
            X_train_selected = X_train[:, extracted_features_idx]
            X_val_selected   = X_val[:, extracted_features_idx]
                
            results, conf_matrix = ridge_regression_fast(
                X_train_selected, labels_train, X_val_selected, labels_val, lambda_value, a_value, u_dc_value
            )

            np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt", 
                       results.reshape(1, -1), fmt="%.5f")
            
            np.savetxt(f"{results_dir}/conf_matrix-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt", 
                       conf_matrix, fmt="%.5f")
