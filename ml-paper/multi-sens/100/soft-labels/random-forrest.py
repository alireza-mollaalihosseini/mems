import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax


def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    A = X_train.T @ X_train + lam * I
    b = X_train.T @ Y_train
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b

def compute_ridge_weights(X_train_std, Y, lam):
    X_train_b = np.hstack((X_train_std, np.ones((X_train_std.shape[0], 1), dtype=X_train_std.dtype)))
    return ridge_closed_form(X_train_b, Y, lam)

def predict_logits(X_std, W):
    X_b = np.hstack((X_std, np.ones((X_std.shape[0], 1), dtype=X_std.dtype)))
    return X_b @ W

def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam,
                          hard_y_train=None, hard_y_eval=None):
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_train.dtype)))

    W = ridge_closed_form(X_train_b, Y_train, lam)

    if hard_y_train is None:
        hard_y_train = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    if hard_y_eval is None:
        hard_y_eval = Y_eval if Y_eval.ndim == 1 else np.argmax(Y_eval, axis=1)

    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == hard_y_train)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(hard_y_eval, y_eval_hats)
    precision = precision_score(hard_y_eval, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(hard_y_eval, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(hard_y_eval, y_eval_hats, average='macro', zero_division=0)
    
    results = np.array([lam, train_accuracy, accuracy, precision, recall, f1])
    
    return results

if __name__ == '__main__':
    lambda_values = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                              1e2, 1e3, 1e4, 1e5, 1e6])
    f_values = np.linspace(1000, 50000, 101)
    num_samples = 10910
    num_tenths = 10

    # Load full 1-second state matrix
    state_matrix_full = np.zeros((num_samples, len(f_values) * 60))
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/f-{int(f)}.npz")["arr_0"]
        state_matrix_full[:, i*60:(i+1)*60] = cols

    # Load 10 sub-segment (0.1 s) state matrices
    sub_state_matrices = []
    for seg in range(1, num_tenths + 1):
        state_mat_seg = np.zeros((num_samples, len(f_values) * 60))
        for i, f in enumerate(f_values):
            cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/one-tenth/f-{int(f)}-{seg}.npz")["arr_0"]
            state_mat_seg[:, i*60:(i+1)*60] = cols
        sub_state_matrices.append(state_mat_seg)

    # Labels (assumed one-hot)
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    hard_y_train = np.argmax(labels_train, axis=1)
    hard_y_val   = np.argmax(labels_val, axis=1)
    hard_y_test  = np.argmax(labels_test, axis=1)

    num_train = labels_train.shape[0]
    num_val   = labels_val.shape[0]

    # Split full and sub matrices
    train_full = state_matrix_full[:num_train]
    val_full   = state_matrix_full[num_train:num_train + num_val]
    test_full  = state_matrix_full[num_train + num_val:]

    sub_train_list = [sub[:num_train] for sub in sub_state_matrices]
    sub_val_list   = [sub[num_train:num_train + num_val] for sub in sub_state_matrices]
    sub_test_list  = [sub[num_train + num_val:] for sub in sub_state_matrices]

    # Train 10 sub-models (teacher ensemble)
    W_list = []
    scaler_sub_list = []
    for k in range(num_tenths):
        print(f"Training sub-model {k+1}/{num_tenths}")
        scaler_k = StandardScaler()
        X_train_k = scaler_k.fit_transform(sub_train_list[k])
        X_val_k   = scaler_k.transform(sub_val_list[k])

        outputs_arr_k = []
        for lam in lambda_values:

            outputs_k = ridge_regression_fast(X_train_k, labels_train, X_val_k, labels_val, lam)
            outputs_arr_k.append(outputs_k)

        outputs_arr_k = np.array(outputs_arr_k)
        best_idx_k = np.argmax(outputs_arr_k[:, 2])
        best_lam_k = outputs_arr_k[best_idx_k, 0]
        best_val_k = outputs_arr_k[best_idx_k, 2]

        print(f"  Best λ = {best_lam_k:.1e}, val accuracy = {best_val_k:.4f}")

        W_k = compute_ridge_weights(X_train_k, labels_train, best_lam_k)
        W_list.append(W_k)
        scaler_sub_list.append(scaler_k)

    # Generate logits from teacher ensemble (no softmax yet)
    logits_train_list = []
    logits_val_list   = []
    logits_test_list  = []
    for k in range(num_tenths):
        logits_train_k = predict_logits(scaler_sub_list[k].transform(sub_train_list[k]), W_list[k])
        logits_train_list.append(logits_train_k)
        
        logits_val_k = predict_logits(scaler_sub_list[k].transform(sub_val_list[k]), W_list[k])
        logits_val_list.append(logits_val_k)
        
        logits_test_k = predict_logits(scaler_sub_list[k].transform(sub_test_list[k]), W_list[k])
        logits_test_list.append(logits_test_k)

    # Proper logit averaging for the teacher ensemble (better than averaging probabilities)
    average_logits_train = np.mean(logits_train_list, axis=0)
    average_logits_val   = np.mean(logits_val_list, axis=0)
    average_logits_test  = np.mean(logits_test_list, axis=0)

    ensemble_val_acc = accuracy_score(hard_y_val, np.argmax(average_logits_val, axis=1))
    ensemble_test_acc = accuracy_score(hard_y_test, np.argmax(average_logits_test, axis=1))
    print(f"\nTeacher ensemble (logit-averaged) val accuracy: {ensemble_val_acc:.4f}")
    print(f"Teacher ensemble (logit-averaged) test accuracy: {ensemble_test_acc:.4f}")


    # === Move scaling of full features OUTSIDE all experiments (same for all configs) ===
    scaler_full = StandardScaler()
    X_train_full_std = scaler_full.fit_transform(train_full)
    X_val_full_std   = scaler_full.transform(val_full)
    X_test_full_std  = scaler_full.transform(test_full)

    print("\n=== Random Forest with soft probability targets (distillation) ===")

    # Use soft probabilities from teacher ensemble (T=1.0 gave best linear result)
    soft_probs_train = softmax(average_logits_train, axis=1)   # n_train x 10

    # Strongly regularized RF (to combat overfitting seen in your hard-label RF)
    rf_soft = RandomForestRegressor(
        n_estimators=1000,          # enough for stability, faster than 5000
        max_depth=20,               # key regularization — prevent deep overfitting
        min_samples_leaf=5,         # avoid tiny leaves
        min_samples_split=10,
        max_features=0.3,           # subsample features heavily in high dim
        bootstrap=True,
        random_state=42,
        n_jobs=64
    )

    rf_soft.fit(X_train_full_std, soft_probs_train)

    # Predict probabilities on val/test
    val_probs_rf = rf_soft.predict(X_val_full_std)
    val_probs_rf = np.clip(val_probs_rf, 0, 1)   # safety
    val_hats_rf = np.argmax(val_probs_rf, axis=1)
    val_acc_rf = accuracy_score(hard_y_val, val_hats_rf)
    print(f"RF-soft val accuracy: {val_acc_rf:.4f}")

    test_probs_rf = rf_soft.predict(X_test_full_std)
    test_probs_rf = np.clip(test_probs_rf, 0, 1)
    test_hats_rf = np.argmax(test_probs_rf, axis=1)

    test_acc_rf = accuracy_score(hard_y_test, test_hats_rf)
    test_prec_rf = precision_score(hard_y_test, test_hats_rf, average='macro', zero_division=0)
    test_rec_rf = recall_score(hard_y_test, test_hats_rf, average='macro', zero_division=0)
    test_f1_rf = f1_score(hard_y_test, test_hats_rf, average='macro', zero_division=0)

    print(f"RF-soft test accuracy:  {test_acc_rf:.4f}")
    print(f"RF-soft test precision: {test_prec_rf:.4f}")
    print(f"RF-soft test recall:    {test_rec_rf:.4f}")
    print(f"RF-soft test F1:        {test_f1_rf:.4f}")

    print("\n=== Regularized Random Forest on hard labels ===")
    rf_hard = RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=0.3,
        random_state=42,
        n_jobs=64,
        class_weight='balanced'
    )
    rf_hard.fit(X_train_full_std, hard_y_train)

    test_hats_hard = rf_hard.predict(X_test_full_std)
    test_acc_hard = accuracy_score(hard_y_test, test_hats_hard)
    print(f"Regularized RF-hard test accuracy: {test_acc_hard:.4f}")