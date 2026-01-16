import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
plt.style.use('ggplot')


def lda_classification_fast(X_train, Y_train, X_eval, Y_eval):
    # -------------------------
    # Handle labels (1D or one-hot)
    # -------------------------
    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    # -------------------------
    # Train LDA
    # -------------------------
    lda = LinearDiscriminantAnalysis(
        solver="svd"   # best default for high-dimensional features
    )
    # lda = LinearDiscriminantAnalysis(
    #     solver="lsqr", shrinkage="auto"   # best default for high-dimensional features
    # ) # worser than svd

    lda.fit(X_train, y_train_true)

    # -------------------------
    # Predictions
    # -------------------------
    y_train_pred = lda.predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train_true)

    y_eval_pred = lda.predict(X_eval)

    accuracy  = accuracy_score(y_eval_true, y_eval_pred)
    precision = precision_score(y_eval_true, y_eval_pred, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_pred, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_pred, average='macro', zero_division=0)

    results = np.array(
        [train_accuracy, accuracy, precision, recall, f1],
        dtype=np.float64
    )

    return results


# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('--a', type=float, required=True, help='Value of a to process')
# parser.add_argument('--u_dc', type=float, required=True, help='Value of u_dc to process')
# args = parser.parse_args()

if __name__ == '__main__':

    # a = args.a
    a = 0.9
    mu = 1.0 
    # u_dc = args.u_dc
    u_dc = 1.0

    # f_values = np.linspace(1000, 50000, 36)
    f_values = np.array([43630, 44120, 42650, 45590,  6390, 44610, 23540, 45100,  6880,
       42160,  2960, 46080, 43140, 24030, 49510, 39220,  3940,  5410,
       38730, 37260,  3450, 25010, 20600, 20110,  5900, 41180,  4430,
        4920, 24520, 47060, 40200, 37750, 41670, 46570, 50000, 39710,
       38240, 36770, 48040, 48530,  7370, 18640, 49020,  2470, 47550,
       40690, 22070, 36280, 23050,  7860, 19620, 18150, 19130, 35300,
       21090, 28930, 21580, 25990,  8840, 11290,  8350,  9330,  1980,
       35790])
    f_values = np.sort(f_values)

    state_matrix = np.zeros((10910, len(f_values) * 60))
    
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*60:(i+1)*60] = cols

        # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)

    # -----------------------------
    # K-Fold Configuration
    # -----------------------------
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(state_matrix), start=1):

        # split
        X_train = state_matrix[train_idx]
        X_test  = state_matrix[test_idx]

        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        # scale PER FOLD (no leakage)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std  = scaler.transform(X_test)

        # evaluate all lambdas (parallel)
        fold_outputs = lda_classification_fast(X_train_std, y_train, X_test_std, y_test)

        results.append(fold_outputs)

        # # Save raw metrics
        # np.savetxt(
        #     f"{results_dir}/fold_results-fold-{fold}.txt",
        #     fold_outputs.reshape(1,-1),
        #     fmt="%.6f"
        # )

    print("\n✅ K-Fold cross-validation finished successfully.")

    # Average results across folds
    results = np.array(results)
    mean_results = np.mean(results, axis=0)

    print("\n=== Final Results ===")
    print(f"Train Accuracy: {mean_results[0]*100:.2f}%")
    print(f"Eval Accuracy:  {mean_results[1]*100:.2f}%")
    print(f"Eval Precision: {mean_results[2]*100:.2f}%")
    print(f"Eval Recall:    {mean_results[3]*100:.2f}%")
    print(f"Eval F1 Score:  {mean_results[4]*100:.2f}%")

    # # delete cols
    # for i, f in enumerate(f_values):
    #     os.remove(f"/scratch/almo2783/scratch/ml-paper/lda/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/f-{int(f)}.npz")

    # err_out_dir = Path(
    #     f"/scratch/almo2783/scratch/ml-paper/lda/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/err-out"
    # )

    # if err_out_dir.exists():
    #     shutil.rmtree(err_out_dir)