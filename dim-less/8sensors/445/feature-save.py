import numpy as np

if __name__ == "__main__":
    # Parameters
    a_value = 0.16
    u_dc_value = 0.1
    alpha = 0.5  # Only process alpha = 6
    mu = 1.0
    lambda_value = 1e4

    # Load data
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
    X_test  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")
    weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/weights/weights-a-{a_value}-lambda-{lambda_value}.npz")

    # load compressed files
    X_train = X_train['arr_0']
    X_test = X_test['arr_0']
    weights = weights['arr_0']

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

    np.save(f"/scratch/almo2783/scratch/dim-less/8sensors/445/selected_features/extracted_features_idx_alpha_{alpha}.npy", extracted_features_idx)

    # if len(extracted_features_idx) != 0:
    #     # Select columns corresponding to the extracted feature indices
    #     X_train_selected = X_train[:, extracted_features_idx]
    #     X_test_selected = X_test[:, extracted_features_idx]

    #     # Save the selected features
    #     save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/selected_features/X_train_selected_alpha_{alpha}.txt", X_train_selected)
    #     save_matrix_to_file(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/selected_features/X_test_selected_alpha_{alpha}.txt", X_test_selected)
