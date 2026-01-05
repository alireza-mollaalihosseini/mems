import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def correlation_filter(X, threshold=0.9):
    """
    Iteratively removes columns that are highly correlated.
    Keeps one representative feature per correlation group.
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)

    # Boolean array to keep track of which columns to keep
    keep = np.ones(corr_matrix.shape[0], dtype=bool)

    for i in range(corr_matrix.shape[0]):
        if keep[i]:
            # Find columns correlated with column i (except itself)
            high_corr = np.where(np.abs(corr_matrix[i, :]) > threshold)[0]
            # Set them to False, except for the current column
            high_corr = high_corr[high_corr > i]
            keep[high_corr] = False

    # Return indices of remaining columns
    selected_indices = np.where(keep)[0]
    return selected_indices


train_X = np.load('/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_X)
idx_80 = correlation_filter(X_scaled, threshold=0.80)
np.save('indices_80.npy', idx_80)
print(f'The length of indices are : {len(idx_80)}')