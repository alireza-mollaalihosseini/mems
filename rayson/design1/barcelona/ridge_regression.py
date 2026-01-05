import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_matrix_from_file(filename):
    return np.loadtxt(filename)

def save_matrix_to_file(filename, matrix):
    np.savetxt(filename, matrix, fmt='%.5f')

def ridge_regression(s_train_data, train_vectors, s_test_data, test_vectors, lam, a, u_dc):

    # Separate the bias column (last column)
    bias_train = s_train_data[:, -1]  # Extract the last column (bias)
    bias_test = s_test_data[:, -1]

    data_train_without_bias = s_train_data[:, :-1]  # Remove the last column
    data_test_without_bias = s_test_data[:, :-1]

    # Standardize the data (excluding the bias column)
    scaler = StandardScaler()
    standardized_train = scaler.fit_transform(data_train_without_bias)
    standardized_test = scaler.transform(data_test_without_bias)

    # Add the bias column back to the standardized data
    s_train_data = np.hstack((standardized_train, bias_train.reshape(-1, 1)))
    s_test_data = np.hstack((standardized_test, bias_test.reshape(-1, 1)))
    
    # Ridge regression model with stronger regularization if needed
    model = Ridge(alpha=lam, fit_intercept=False, solver="cholesky")  # Using 'cholesky' for better stability
    model.fit(s_train_data, train_vectors)
    
    # Training predictions and accuracy
    y_train_pred = model.predict(s_train_data)
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = accuracy_score(np.argmax(train_vectors, axis=1), y_train_hats)
    
    # Testing predictions and accuracy
    y_test_pred = model.predict(s_test_data)
    y_test_hats = np.argmax(y_test_pred, axis=1)
    test_accuracy = accuracy_score(np.argmax(test_vectors, axis=1), y_test_hats)
    
    # Evaluation metrics (precision, recall, f1)
    accuracy = accuracy_score(np.argmax(test_vectors, axis=1), y_test_hats)
    precision = precision_score(np.argmax(test_vectors, axis=1), y_test_hats, average='macro', zero_division=0)
    recall = recall_score(np.argmax(test_vectors, axis=1), y_test_hats, average='macro', zero_division=0)
    f1 = f1_score(np.argmax(test_vectors, axis=1), y_test_hats, average='macro', zero_division=0)
    
    # Confusion matrix
    all_predictions = list(y_test_hats)
    all_true_labels = list(np.argmax(test_vectors, axis=1))
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    # Store results
    results = np.array([a, u_dc, lam, train_accuracy, test_accuracy, accuracy, precision, recall, f1])
    
    return model, scaler, results, conf_matrix


if __name__ == "__main__":
    # Parse command-line arguments
    a_value = float(sys.argv[1])
    u_dc_value = float(sys.argv[2])
    lambda_value = float(sys.argv[3])

    X_train = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/X_train_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    X_val   = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/X_val_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    X_test  = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/X_test_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    y_train = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/y_train_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    y_val   = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/y_val_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    y_test  = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/y_test_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")

    # Measure training time
    start_train = time.time()
    model, scaler, results, conf_matrix_val = ridge_regression(X_train, y_train, X_val, y_val, lambda_value, a_value, u_dc_value)
    end_train = time.time()
    training_time = end_train - start_train
    print(f"Training time: {training_time:.4f} seconds")

    # Save the results (scalars)
    save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/results/results-a-{a_value:.2f}-u_dc-{u_dc_value:.1f}.txt", results)
    
    # Save confusion matrix (2D array)
    save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/results/conf_matrix-a-{a_value:.2f}-u_dc-{u_dc_value:.1f}.txt", conf_matrix_val)


    # -------------------------------
    # Final Evaluation on Independent Test Data
    # -------------------------------
    
    # Process test data using the scaler fitted on the training data
    bias_test = X_test[:, -1]
    data_test = X_test[:, :-1]
    standardized_test = scaler.transform(data_test)
    s_test = np.hstack((standardized_test, bias_test.reshape(-1, 1)))
    
    # Measure inference time
    start_inference = time.time()    
    y_test_pred = model.predict(s_test)
    y_test_hats = np.argmax(y_test_pred, axis=1)
    true_test_labels = np.argmax(y_test, axis=1)
    end_inference = time.time()
    inference_time = end_inference - start_inference
    print(f"Inference time: {inference_time:.4f} seconds")
    
    test_accuracy = accuracy_score(true_test_labels, y_test_hats)
    precision_test = precision_score(true_test_labels, y_test_hats, average='macro', zero_division=0)
    recall_test = recall_score(true_test_labels, y_test_hats, average='macro', zero_division=0)
    f1_test = f1_score(true_test_labels, y_test_hats, average='macro', zero_division=0)
    conf_matrix_test = confusion_matrix(true_test_labels, y_test_hats)
    
    # Pack final test results: [a, u_dc, lambda, test_accuracy, precision, recall, f1]
    test_results = np.array([a_value, u_dc_value, lambda_value, test_accuracy, precision_test, recall_test, f1_test])

    # Save the results (scalars)
    save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/results/results-a-{a_value:.2f}-u_dc-{u_dc_value:.1f}-test.txt", test_results)
    
    # Save confusion matrix (2D array)
    save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/results/conf_matrix-a-{a_value:.2f}-u_dc-{u_dc_value:.1f}-test.txt", conf_matrix_test)