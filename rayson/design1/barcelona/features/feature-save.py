import numpy as np

def load_matrix_from_file(filename):
    return np.loadtxt(filename)

def save_matrix_to_file(filename, matrix):
    np.savetxt(filename, matrix, fmt='%.5f')

if __name__ == "__main__":
    # Parameters
    a_value = -1.08
    u_dc_value = 0.1
    alpha = 6  # Only process alpha = 6

    # Load data
    X_train = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/X_train_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    X_test = load_matrix_from_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/state-matrices/X_test_a_{a_value:.2f}_u_dc_{u_dc_value:.1f}.txt")
    weights = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/weights/weights-a--1.08-lambda-10000.0.txt")

    # Extracted feature indices
    extracted_features_idx = []

    # Iterate over the classes
    for i in range(10):
        positive_part = [x for x in weights[i] if x > 0]
        negative_part = [x for x in weights[i] if x < 0]

        positive_median = np.median(positive_part) if len(positive_part) > 0 else 0
        negative_median = np.median(negative_part) if len(negative_part) > 0 else 0

        extracted_features_positive_idx = np.where(weights[i] > alpha * positive_median)[0]
        extracted_features_negative_idx = np.where(weights[i] < alpha * negative_median)[0]

        extracted_features_idx.extend(extracted_features_positive_idx)
        extracted_features_idx.extend(extracted_features_negative_idx)

    # Remove duplicate indices
    extracted_features_idx = np.unique(extracted_features_idx)

    print(f"The Length of the Extracted Feature Indices with alpha = {alpha}: {len(extracted_features_idx)}")

    save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/selected_features/extracted_features_idx_alpha_{alpha}.txt", extracted_features_idx)

    # if len(extracted_features_idx) != 0:
    #     # Select columns corresponding to the extracted feature indices
    #     X_train_selected = X_train[:, extracted_features_idx]
    #     X_test_selected = X_test[:, extracted_features_idx]

    #     # Save the selected features
    #     save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/selected_features/X_train_selected_alpha_{alpha}.txt", X_train_selected)
    #     save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/selected_features/X_test_selected_alpha_{alpha}.txt", X_test_selected)
