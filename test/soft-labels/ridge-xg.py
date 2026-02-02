import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


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


# === Helper function for distillation + evaluation ===
def distill_and_eval(soft_Y_train, name, tolerance=0.02):
    print(f"\n=== {name} ===")
    
    outputs_arr = []
    for lam in lambda_values:
        outputs = ridge_regression_fast(
            X_train_full_std, soft_Y_train,
            X_val_full_std, labels_val,  # labels_val is one-hot, used for hard acc inside function
            lam,
            hard_y_train=hard_y_train,
            hard_y_eval=hard_y_val
        )
        outputs_arr.append(outputs)
    
    outputs_arr = np.array(outputs_arr)
    
    max_val_acc = np.max(outputs_arr[:, 2])
    candidates = outputs_arr[outputs_arr[:, 2] >= max_val_acc - tolerance]
    best_idx = np.argmax(candidates[:, 0])  # prefer highest lambda
    best_lam = candidates[best_idx, 0]
    best_val = candidates[best_idx, 2]
    best_train = candidates[best_idx, 1]
    
    print(f"Best λ = {best_lam:.1e} (conservative, tol={tolerance}), Train acc = {best_train:.4f}, val acc = {best_val:.4f}")
    
    # Final test evaluation
    W_final = compute_ridge_weights(X_train_full_std, soft_Y_train, best_lam)
    test_logits = predict_logits(X_test_full_std, W_final)
    test_hats = np.argmax(test_logits, axis=1)
    
    test_acc = accuracy_score(hard_y_test, test_hats)
    test_prec = precision_score(hard_y_test, test_hats, average='macro', zero_division=0)
    test_rec = recall_score(hard_y_test, test_hats, average='macro', zero_division=0)
    test_f1 = f1_score(hard_y_test, test_hats, average='macro', zero_division=0)
    
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test precision: {test_prec:.4f}")
    print(f"Test recall:    {test_rec:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    
    return test_acc


if __name__ == '__main__':
    lambda_values = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                              1e2, 1e3, 1e4, 1e5, 1e6])
    # f_values = np.linspace(1000, 50000, 101)
    f_values = np.sort(np.array([43630, 44120, 42650, 45590,  6390, 44610, 23540, 45100,  6880,
       42160,  2960, 46080, 43140, 24030, 49510, 39220,  3940,  5410,
       38730, 37260,  3450, 25010, 20600, 20110,  5900, 41180,  4430,
        4920, 24520, 47060, 40200, 37750, 41670, 46570, 50000, 39710,
       38240, 36770, 48040, 48530,  7370, 18640, 49020,  2470, 47550,
       40690, 22070, 36280, 23050,  7860, 19620, 18150, 19130, 35300,
       21090, 28930, 21580, 25990,  8840, 11290,  8350,  9330,  1980,
       35790]))
    num_samples = 10910
    num_tenths = 10

    # load scores from gini
    ranked_idx = np.load("/scratch/almo2783/scratch/ml-paper/multi-sens/100/top_64/gini/feature_ranking_idx.npy")
    rf_scores = np.load("/scratch/almo2783/scratch/ml-paper/multi-sens/100/top_64/gini/feature_importances.npy")
    top_k = 680
    selected_idx = ranked_idx[:top_k]

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

    # selected features
    train_full = train_full[:, selected_idx]
    val_full   = val_full[:, selected_idx]
    test_full  = test_full[:, selected_idx]

    # sub_train_list = [sub[:num_train] for sub in sub_state_matrices]
    # sub_val_list   = [sub[num_train:num_train + num_val] for sub in sub_state_matrices]
    # sub_test_list  = [sub[num_train + num_val:] for sub in sub_state_matrices]

    sub_train_list = [sub[:num_train, selected_idx] for sub in sub_state_matrices]
    sub_val_list   = [sub[num_train:num_train + num_val, selected_idx] for sub in sub_state_matrices]
    sub_test_list  = [sub[num_train + num_val:, selected_idx] for sub in sub_state_matrices]

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
        best_train_k = outputs_arr_k[best_idx_k, 1]

        print(f"  Best λ = {best_lam_k:.1e}, Train accuracy = {best_train_k:.4f}, val accuracy = {best_val_k:.4f}")

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

    ensemble_train_acc = accuracy_score(hard_y_train, np.argmax(average_logits_train, axis=1))
    ensemble_val_acc = accuracy_score(hard_y_val, np.argmax(average_logits_val, axis=1))
    ensemble_test_acc = accuracy_score(hard_y_test, np.argmax(average_logits_test, axis=1))
    print(f"\nTeacher ensemble (logit-averaged) train accuracy: {ensemble_train_acc:.4f}")
    print(f"Teacher ensemble (logit-averaged) val accuracy: {ensemble_val_acc:.4f}")
    print(f"Teacher ensemble (logit-averaged) test accuracy: {ensemble_test_acc:.4f}")


    # === Move scaling of full features OUTSIDE all experiments (same for all configs) ===
    scaler_full = StandardScaler()
    X_train_full_std = scaler_full.fit_transform(train_full)
    X_val_full_std   = scaler_full.transform(val_full)
    X_test_full_std  = scaler_full.transform(test_full)

    print("\n=== XGBoost with soft probability distillation (T=1.0 teacher targets) ===")

    # Soft targets from teacher ensemble (T=1.0 gave best linear results)
    soft_probs_train = softmax(average_logits_train, axis=1)  # n_train x 10

    # XGBoost regressor wrapped for multi-output
    xgb_soft = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=1500,         # more trees for stability
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            n_jobs=64,
            random_state=42,
            tree_method='hist'         # faster on CPU
        )
    )

    xgb_soft.fit(X_train_full_std, soft_probs_train)

    # Predict probabilities
    val_probs_xgb = np.clip(xgb_soft.predict(X_val_full_std), 0, 1)
    val_hats_xgb = np.argmax(val_probs_xgb, axis=1)
    print(f"XGBoost-soft val accuracy: {accuracy_score(hard_y_val, val_hats_xgb):.4f}")

    test_probs_xgb = np.clip(xgb_soft.predict(X_test_full_std), 0, 1)
    test_hats_xgb = np.argmax(test_probs_xgb, axis=1)

    test_acc_xgb = accuracy_score(hard_y_test, test_hats_xgb)
    test_prec_xgb = precision_score(hard_y_test, test_hats_xgb, average='macro', zero_division=0)
    test_rec_xgb = recall_score(hard_y_test, test_hats_xgb, average='macro', zero_division=0)
    test_f1_xgb = f1_score(hard_y_test, test_hats_xgb, average='macro', zero_division=0)

    print(f"XGBoost-soft test accuracy:  {test_acc_xgb:.4f}")
    print(f"XGBoost-soft test precision: {test_prec_xgb:.4f}")
    print(f"XGBoost-soft test recall:    {test_rec_xgb:.4f}")
    print(f"XGBoost-soft test F1:        {test_f1_xgb:.4f}")

    print("\n=== Ultimate Stacking: Linear Student (T=1.0) + Teacher Ensemble + XGBoost-soft ===")

    # 1. Linear student probs (recompute if not saved)
    best_T = 1.0
    best_lam_lin = 1e2  # from your conservative T=1.0 run
    soft_Y_lin = softmax(average_logits_train / best_T, axis=1)
    W_lin = compute_ridge_weights(X_train_full_std, soft_Y_lin, best_lam_lin)
    lin_logits_test = predict_logits(X_test_full_std, W_lin)
    lin_probs_test = softmax(lin_logits_test, axis=1)

    # 2. Teacher ensemble probs
    teacher_probs_test = softmax(average_logits_test, axis=1)

    # 3. XGBoost-soft probs (from your run above)
    xgb_probs_test = np.clip(xgb_soft.predict(X_test_full_std), 0, 1)  # assuming xgb_soft from previous block

    # Weighted average — equal weights first (often best), or tune slightly if one is stronger
    ultimate_probs_test = (lin_probs_test + teacher_probs_test + xgb_probs_test) / 3.0
    ultimate_hats = np.argmax(ultimate_probs_test, axis=1)

    ultimate_acc = accuracy_score(hard_y_test, ultimate_hats)
    ultimate_prec = precision_score(hard_y_test, ultimate_hats, average='macro', zero_division=0)
    ultimate_rec = recall_score(hard_y_test, ultimate_hats, average='macro', zero_division=0)
    ultimate_f1 = f1_score(hard_y_test, ultimate_hats, average='macro', zero_division=0)

    print(f"Ultimate stacked test accuracy:  {ultimate_acc:.4f}")
    print(f"Ultimate stacked test precision: {ultimate_prec:.4f}")
    print(f"Ultimate stacked test recall:    {ultimate_rec:.4f}")
    print(f"Ultimate stacked test F1:        {ultimate_f1:.4f}")