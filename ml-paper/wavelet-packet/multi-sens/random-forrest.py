import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
plt.style.use('ggplot')


# def rf_classification_fast(X_train, Y_train, X_eval, Y_eval, n_estimators=500, max_depth=None, min_samples_leaf=1):
#     y_train_true = np.argmax(Y_train, axis=1) if Y_train.ndim > 1 else Y_train
#     y_eval_true  = np.argmax(Y_eval, axis=1)  if Y_eval.ndim > 1   else Y_eval

#     rf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_leaf=min_samples_leaf,
#         n_jobs=64,
#         random_state=42,
#         class_weight='balanced'
#     )
#     rf.fit(X_train, y_train_true)

#     train_acc = rf.score(X_train, y_train_true)
#     val_pred = rf.predict(X_eval)
#     val_acc = accuracy_score(y_eval_true, val_pred)
#     precision = precision_score(y_eval_true, val_pred, average='macro', zero_division=0)
#     recall = recall_score(y_eval_true, val_pred, average='macro', zero_division=0)
#     f1 = f1_score(y_eval_true, val_pred, average='macro', zero_division=0)

#     return np.array([n_estimators, max_depth, min_samples_leaf, train_acc, val_acc, precision, recall, f1])

def rf_classification_fast(X_train, Y_train, X_eval, Y_eval,
                           n_estimators=500, max_depth=None, min_samples_leaf=1,
                           max_features='sqrt', class_weight=None):
    y_train_true = np.argmax(Y_train, axis=1) if Y_train.ndim > 1 else Y_train
    y_eval_true  = np.argmax(Y_eval, axis=1)  if Y_eval.ndim > 1   else Y_eval

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=64,
        random_state=42
    )
    rf.fit(X_train, y_train_true)

    train_acc = rf.score(X_train, y_train_true)
    val_pred = rf.predict(X_eval)
    val_acc = accuracy_score(y_eval_true, val_pred)
    precision = precision_score(y_eval_true, val_pred, average='macro', zero_division=0)
    recall = recall_score(y_eval_true, val_pred, average='macro', zero_division=0)
    f1 = f1_score(y_eval_true, val_pred, average='macro', zero_division=0)

    return np.array([
        n_estimators, max_depth, min_samples_leaf,
        max_features if isinstance(max_features, str) else float(max_features),
        '-' if class_weight is None else 'balanced',
        train_acc, val_acc, precision, recall, f1
    ], dtype=object)


if __name__ == '__main__':

    # a = args.a
    a = 0.9
    mu = 1.0 
    # u_dc = args.u_dc
    u_dc = 1.0
    
    f_values = np.linspace(1000, 50000, 36)
    # param_grid = [
    #     (300, None, 1),
    #     (500, None, 1),
    #     (800, None, 1),
    #     (500, 30, 1),
    #     (500, None, 5)
    # ]
    param_grid = [
        (800, None, 1, 'sqrt', None),
        (800, None, 1, 0.3, None),
        (800, None, 1, 0.5, None),
        (1000, None, 1, 'sqrt', None),
        (800, None, 2, 'sqrt', None),
        (800, None, 1, 'sqrt', 'balanced'),
        (1200, None, 1, 'sqrt', None),
        (1500, None, 1, 'sqrt', None)
    ]

    state_matrix = np.zeros((10510, len(f_values) * 368))
    
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*368:(i+1)*368] = cols

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    train_state = state_matrix[:len(labels_train)]
    val_state   = state_matrix[len(labels_train):]

    # scale PER FOLD (no leakage)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_state)
    X_val_std   = scaler.transform(val_state)

    # evaluate all lambdas (parallel)
    outputs = Parallel(
        n_jobs=64,
        verbose=1,
        backend="multiprocessing"
    )(
        delayed(rf_classification_fast)(
            X_train_std, labels_train,
            X_val_std,  labels_val,
            # n_est, m_depth, msl
            n_est, m_depth, msl, mf, cw
        )
        # for n_est, m_depth, msl in param_grid
        for n_est, m_depth, msl, mf, cw in param_grid
    )

    outputs_arr = np.vstack(outputs)

    max_idx = np.argmax(outputs_arr[:, 6])
    best_val_acc = outputs_arr[max_idx, 6]
    best_params = outputs_arr[max_idx, :5]

    print(f"\nOverall best val_acc: {best_val_acc * 100:.6f} % with Random Forrest params: {best_params}")

    # # Results dir
    # results_dir = f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/accuracy"
    # os.makedirs(results_dir, exist_ok=True)

    # # Save raw metrics
    # np.savetxt(
    #     f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-{mu:.2e}-rf.txt",
    #     outputs_arr,
    #     fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
    #     header="n_estimators,max_depth,min_samples_leaf,train_acc,val_acc,precision,recall,f1",
    #     comments=""
    # )

    # # plot the results
    # lambdas   = outputs_arr[:, 0]
    # train_acc = outputs_arr[:, 1] * 100
    # test_acc  = outputs_arr[:, 2] * 100

    # idx_best = np.argmax(test_acc)

    # best_test   = test_acc[idx_best]
    # best_train  = train_acc[idx_best]
    # best_lambda = lambdas[idx_best]

    # # std across λ (for uncertainty band)
    # std_test = np.std(test_acc)
    # std_train = np.std(train_acc)

    # # -----------------------------
    # # Plot with error bars
    # # -----------------------------
    # plt.figure(figsize=(16, 8))

    # plt.plot(
    #     lambda_values, test_acc,
    #     marker='o',
    #     linewidth=3,
    #     label="Validation Accuracy"
    # )

    # plt.plot(
    #     lambda_values, train_acc,
    #     marker='s',
    #     linewidth=3,
    #     linestyle='--',
    #     label="Training Accuracy"
    # )

    # # highlight best lambda
    # plt.scatter(
    #     best_lambda,
    #     best_test,
    #     s=200,
    #     marker='*',
    #     zorder=5,
    #     label=(
    #         f"Best val = {best_test:.2f}%\n"
    #         f"λ = {best_lambda:.1e}"
    #     )
    # )

    # # Formatting
    # plt.xscale('log')
    # plt.xlabel("Ridge $\\lambda$", fontweight='bold', fontsize=20)
    # plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
    # plt.xticks(fontweight='bold', fontsize=20)
    # plt.yticks(fontweight='bold', fontsize=20)
    # plt.legend(fontsize=18)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    # plt.tight_layout()

    # plt.savefig(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/lambda-optimization-a-{a:.2f}-u_dc-{u_dc:.2f}.png", dpi=300)
    # plt.close()

    # # Results dir
    # results_dir = f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/accuracy"
    # os.makedirs(results_dir, exist_ok=True)

    # # Save raw metrics
    # np.savetxt(
    #     f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-{mu:.2e}.txt",
    #     outputs_arr,
    #     fmt=["%.0e", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
    #     header="lambda,train_acc,val_acc,precision,recall,f1",
    #     comments=""
    # )

    # # # delete cols
    # # for i, f in enumerate(f_values):
    # #     os.remove(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/f-{int(f)}.npz")

    # # err_out_dir = Path(
    # #     f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/err-out"
    # # )

    # # if err_out_dir.exists():
    # #     shutil.rmtree(err_out_dir)