import sys
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_matrix_from_file(filename):
    return np.loadtxt(filename)

def save_matrix_to_file(filename, matrix):
    np.savetxt(filename, matrix, fmt='%.5f')

def ridge_regression(s_train_data, train_vectors, s_test_data, test_vectors, lam, a, u_dc):
    # Separate the bias column (last column)
    bias_train = s_train_data[:, -1]   # Extract the last column (bias)
    bias_test  = s_test_data[:, -1]

    data_train_without_bias = s_train_data[:, :-1]  # Remove the last column
    data_test_without_bias  = s_test_data[:, :-1]

    # Standardize the data (excluding the bias column)
    scaler = StandardScaler()
    standardized_train = scaler.fit_transform(data_train_without_bias)
    standardized_test  = scaler.transform(data_test_without_bias)

    # Re-attach the bias column
    s_train_data = np.hstack((standardized_train, bias_train.reshape(-1, 1)))
    s_test_data  = np.hstack((standardized_test, bias_test.reshape(-1, 1)))
    
    # Ridge regression model (using the closed-form via scikit-learn's implementation)
    # Measure training time
    start_train = time.time()
    model = Ridge(alpha=lam, fit_intercept=False, solver="cholesky")
    model.fit(s_train_data, train_vectors)
    
    # Training predictions and accuracy
    y_train_pred = model.predict(s_train_data)
    y_train_hats = np.argmax(y_train_pred, axis=1)
    end_train = time.time()
    training_time = end_train - start_train
    
    train_accuracy = accuracy_score(np.argmax(train_vectors, axis=1), y_train_hats)
    
    # Testing predictions and accuracy
    # Measure inference time
    start_inference = time.time()
    y_test_pred = model.predict(s_test_data)
    y_test_hats = np.argmax(y_test_pred, axis=1)
    end_inference = time.time()
    inference_time = end_inference - start_inference

    test_accuracy = accuracy_score(np.argmax(test_vectors, axis=1), y_test_hats)
    
    # Evaluation metrics (precision, recall, f1)
    acc = accuracy_score(np.argmax(test_vectors, axis=1), y_test_hats)
    precision = precision_score(np.argmax(test_vectors, axis=1), y_test_hats, average='macro', zero_division=0)
    recall = recall_score(np.argmax(test_vectors, axis=1), y_test_hats, average='macro', zero_division=0)
    f1 = f1_score(np.argmax(test_vectors, axis=1), y_test_hats, average='macro', zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(np.argmax(test_vectors, axis=1), y_test_hats)
    
    # Store results: [a, u_dc, lambda, train_accuracy, test_accuracy, accuracy, precision, recall, f1]
    results = np.array([a, u_dc, lam, train_accuracy, test_accuracy, acc, precision, recall, f1])
    
    return results, conf_matrix, training_time, inference_time


if __name__ == "__main__":
    # Parse command-line arguments
    a_value = -1.08
    u_dc_value = 0.1
    lambda_value = 1e4
    alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
    lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

    # Load training, validation, and testing state matrices and label matrices
    X_train = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/state-matrices/X_train_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    # X_val   = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/state-matrices/X_val_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    X_test  = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/state-matrices/X_test_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    y_train = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/state-matrices/y_train_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    # y_val   = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/state-matrices/y_val_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    y_test  = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/state-matrices/y_test_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")

    # Load the weights
    weights = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/5cities/features/weights/weights-a--1.08-lambda-10000.0.txt")

    for alpha in alpha_values:

        # List to store extracted feature indices
        extracted_features_idx = []

        # Iterate over the classes
        for i in range(10):
            # Split data into positive and negative parts
            positive_part = [x for x in weights[i] if x > 0]
            negative_part = [x for x in weights[i] if x < 0]

            # Indices where the parts are positive or negative
            positive_part_idx = np.where(weights[i] > 0)[0]
            negative_part_idx = np.where(weights[i] < 0)[0]

            # Calculate median of the positive and negative parts
            positive_median = np.median(positive_part) if len(positive_part) > 0 else 0
            negative_median = np.median(negative_part) if len(negative_part) > 0 else 0

            # Indices where values are greater/less than the median
            extracted_features_positive_idx = np.where(weights[i] > alpha * positive_median)[0]
            extracted_features_negative_idx = np.where(weights[i] < alpha * negative_median)[0]

            # Append indices to the list
            extracted_features_idx.extend(extracted_features_positive_idx)
            extracted_features_idx.extend(extracted_features_negative_idx)

        # Remove duplicate indices using np.unique
        extracted_features_idx = np.unique(extracted_features_idx)

        # Print the result
        print(f"The Length of the Extracted Feature Indices with alpha = {int(alpha)}: {int(len(extracted_features_idx))}")

        if len(extracted_features_idx) != 0:

            # Select columns corresponding to the extracted feature indices
            X_train_selected = X_train[:, extracted_features_idx]
            X_test_selected  = X_test[:, extracted_features_idx]


            for lambda_value in lambda_values:

                results, conf_matrix, training_time, inference_time = ridge_regression(X_train_selected, y_train, X_test_selected, y_test, lambda_value, a_value, u_dc_value)
                
                print(f"Training time for alpha={int(alpha)} and lambda={lambda_value:.3e}: {training_time:.4f} seconds")
                print(f"Inference time : {inference_time:.4f} seconds")

                # Save the results (scalars)
                save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/features/results/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt", results)
                
                # Save confusion matrix (2D array)
                save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/5cities/features/results/conf_matrix-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt", conf_matrix)