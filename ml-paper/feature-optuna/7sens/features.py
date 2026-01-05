import numpy as np
import optuna


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


# Load weights
weights_445  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/weights/weights-a-0.16-lambda-10000.0.npz")['arr_0']
weights_490  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/weights/weights-a-0.04-lambda-10000.0.npz")['arr_0']
weights_582  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/weights/weights-a-0.02-lambda-10000.0.npz")['arr_0']
weights_591  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/weights/weights-a-0.6-lambda-10000.0.npz")['arr_0']
weights_1109 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/weights/weights-a-0.06-lambda-10000.0.npz")['arr_0']
weights_1161 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']
weights_2600 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']

results_dir = '/scratch/almo2783/scratch/ml-paper/feature-optuna/results'

study_445  = optuna.load_study(study_name="ridge_feature_selection_weights_445", storage=f"sqlite:///{results_dir}/optuna-weights-445.db")
study_490  = optuna.load_study(study_name="ridge_feature_selection_weights_490", storage=f"sqlite:///{results_dir}/optuna-weights-490.db")
study_582  = optuna.load_study(study_name="ridge_feature_selection_weights_582", storage=f"sqlite:///{results_dir}/optuna-weights-582.db")
study_591  = optuna.load_study(study_name="ridge_feature_selection_weights_591", storage=f"sqlite:///{results_dir}/optuna-weights-591.db")
study_1109 = optuna.load_study(study_name="ridge_feature_selection_weights_1109", storage=f"sqlite:///{results_dir}/optuna-weights-1109.db")
study_1161 = optuna.load_study(study_name="ridge_feature_selection_weights_1161", storage=f"sqlite:///{results_dir}/optuna-weights-1161.db")
study_2600 = optuna.load_study(study_name="ridge_feature_selection_weights", storage=f"sqlite:///{results_dir}/optuna-weights.db")

best_params_445  = study_445.best_params
best_params_490  = study_490.best_params
best_params_582  = study_582.best_params
best_params_591  = study_591.best_params
best_params_1109 = study_1109.best_params
best_params_1161 = study_1161.best_params
best_params_2600 = study_2600.best_params

selected_features_445  = top_features(weights_445, top_k=min(best_params_445['n_features'], weights_445.shape[0] - 1)).tolist()
selected_features_490  = top_features(weights_490, top_k=min(best_params_490['n_features'], weights_490.shape[0] - 1)).tolist()
selected_features_582  = top_features(weights_582, top_k=min(best_params_582['n_features'], weights_582.shape[0] - 1)).tolist()
selected_features_591  = top_features(weights_591, top_k=min(best_params_591['n_features'], weights_591.shape[0] - 1)).tolist()
selected_features_1109 = top_features(weights_1109, top_k=min(best_params_1109['n_features'], weights_1109.shape[0] - 1)).tolist()
selected_features_1161 = top_features(weights_1161, top_k=min(best_params_1161['n_features'], weights_1161.shape[0] - 1)).tolist()
selected_features_2600 = top_features(weights_2600, top_k=min(best_params_2600['n_features'], weights_2600.shape[0] - 1)).tolist()

np.save(f'{results_dir}/selected_features_445.npy', selected_features_445)
np.save(f'{results_dir}/selected_features_490.npy', selected_features_490)
np.save(f'{results_dir}/selected_features_582.npy', selected_features_582)
np.save(f'{results_dir}/selected_features_591.npy', selected_features_591)
np.save(f'{results_dir}/selected_features_1109.npy', selected_features_1109)
np.save(f'{results_dir}/selected_features_1161.npy', selected_features_1161)
np.save(f'{results_dir}/selected_features_2600.npy', selected_features_2600)