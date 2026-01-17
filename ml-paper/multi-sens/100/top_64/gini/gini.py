import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


a_value = 0.9
u_dc_value = 1.0
mu = 1.0

# f_values = np.linspace(1000, 50000, 101)
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

X_train = state_matrix[:len(labels_train)]
X_val   = state_matrix[len(labels_train): len(labels_train) + len(labels_val)]
X_test  = state_matrix[len(labels_train) + len(labels_val): ]

# Convert labels to class indices (0..9)
y_train = np.argmax(labels_train, axis=1) if labels_train.ndim > 1 else labels_train
y_val   = np.argmax(labels_val, axis=1) if labels_val.ndim > 1 else labels_val
y_test  = np.argmax(labels_test, axis=1) if labels_test.ndim > 1 else labels_test

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

rf = RandomForestClassifier(
        n_estimators=5000,          # sweet spot: max stability without insanity
        max_depth=None,             # let trees grow fully (important for pure Gini importance)
        min_samples_leaf=1,         # no regularization → purest possible importance
        min_samples_split=2,
        max_features=0.5,           # or 'sqrt' — 0.5 often gives slightly more stable rankings in very high dim
        bootstrap=True,             # must be True for proper out-of-bag based importance
        oob_score=True,             # optional sanity check
        random_state=42,
        n_jobs=64
    )
rf.fit(X_train, y_train)

# Predict on validation set
y_pred = rf.predict(X_val)

acc = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')

# Validation results
print("Validation Results:")
print(f"Validation accuracy: {acc:.4f}")
print(f"Precision (macro):   {precision:.4f}")
print(f"Recall (macro):      {recall:.4f}")
print(f"F1-score (macro):    {f1:.4f}")

# Extract feature importances
rf_scores = rf.feature_importances_
ranked_idx = np.argsort(rf_scores)[::-1]  # descending

# Save model and preprocessing
dump(rf, "rf_classifier.joblib")
dump(scaler, "scaler.joblib")

# Save feature ranking (already done, but grouped for clarity)
np.save("feature_ranking_idx.npy", ranked_idx)
np.save("feature_importances.npy", rf_scores)

# # Optional: save metrics
# np.save(
#     "validation_metrics.npy",
#     accuracy=acc,
#     precision=precision,
#     recall=recall,
#     f1=f1
# )

# Predict on test set
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Test results
print("Test Results:")
print(f"Test accuracy: {acc:.4f}")
print(f"Precision (macro):   {precision:.4f}")
print(f"Recall (macro):      {recall:.4f}")
print(f"F1-score (macro):    {f1:.4f}")