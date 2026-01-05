# import os
# import numpy as np
# import soundfile as sf
# from joblib import Parallel, delayed


# def process_file(fname):
#     data, _ = sf.read(fname)
#     data = data.astype(np.float32)

#     # DC removal
#     data -= np.mean(data)

#     fft_vals = np.fft.rfft(data)
#     return np.log10(np.abs(fft_vals)+1e-16).astype(np.float32)


# def build_state_matrices(train_file_list_path,
#                          val_file_list_path,
#                          test_file_list_path):

#     train_filenames = np.loadtxt(train_file_list_path, dtype=str)
#     val_filenames   = np.loadtxt(val_file_list_path, dtype=str)
#     test_filenames  = np.loadtxt(test_file_list_path, dtype=str)

#     all_filenames = np.concatenate([
#         train_filenames,
#         val_filenames,
#         test_filenames
#     ])

#     results = Parallel(
#         n_jobs=64,
#         backend="threading",
#         verbose=1
#     )(
#         delayed(process_file)(fname)
#         for fname in all_filenames
#     )

#     state_matrix = np.vstack(results)
#     return state_matrix


# if __name__ == "__main__":

#     # --------------------------------------------------
#     # Paths
#     # --------------------------------------------------
#     train_files = "/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv"
#     val_files   = "/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv"
#     test_files  = "/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv"

#     # OPTIONAL: labels only for evaluation
#     labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
#     labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
#     labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
#     labels = np.concatenate([labels_train, labels_val, labels_test])

#     # --------------------------------------------------
#     # Feature Extraction
#     # --------------------------------------------------
#     print("Building state matrix...")
#     state_matrix = build_state_matrices(
#         train_files,
#         val_files,
#         test_files
#     )

#     np.savez_compressed("/scratch/almo2783/scratch/audio-reg/state-matrix/state_matrix.npz", state_matrix)



import os
import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed


def extract_logmel(
    data,
    sr,
    n_mels=64,
    n_fft=2048,
    hop_length=512
):
    """
    Returns fixed-length log-mel feature:
    [mean(mel), std(mel)] → shape = (2 * n_mels,)
    """

    mel = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    logmel = np.log(mel + 1e-8)

    # Temporal pooling
    feat_mean = np.mean(logmel, axis=1)
    feat_std  = np.std(logmel, axis=1)

    return np.concatenate([feat_mean, feat_std], axis=0)


def extract_mfcc_deltas(
    data,
    sr,
    n_mfcc=20,
    n_fft=2048,
    hop_length=512
):
    """
    Returns fixed-length MFCC feature:
    [MFCC, Δ, ΔΔ] × [mean, std]
    → shape = (3 * 2 * n_mfcc,)
    """

    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def pool(x):
        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=0)

    feat = np.concatenate([
        pool(mfcc),
        pool(delta),
        pool(delta2)
    ], axis=0)

    return feat



def process_file(
    fname,
    feature_type="logmel",  # "logmel" or "mfcc"
):
    data, sr = sf.read(fname)
    data = data.astype(np.float32)

    # Mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # DC removal
    data -= np.mean(data)

    if feature_type == "logmel":
        return extract_logmel(data, sr).astype(np.float32)

    elif feature_type == "mfcc":
        return extract_mfcc_deltas(data, sr).astype(np.float32)

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def build_state_matrices(
    train_file_list_path,
    val_file_list_path,
    test_file_list_path,
    feature_type="logmel"
):
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames   = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames  = np.loadtxt(test_file_list_path, dtype=str)

    all_filenames = np.concatenate([
        train_filenames,
        val_filenames,
        test_filenames
    ])

    results = Parallel(
        n_jobs=64,
        backend="multiprocessing",
        verbose=1
    )(
        delayed(process_file)(fname, feature_type)
        for fname in all_filenames
    )

    state_matrix = np.vstack(results)
    return state_matrix


if __name__ == "__main__":

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    train_files = "/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv"
    val_files   = "/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv"
    test_files  = "/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv"

    # OPTIONAL: labels only for evaluation
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels = np.concatenate([labels_train, labels_val, labels_test])

    # --------------------------------------------------
    # State Matrix
    # --------------------------------------------------
    print("Building state matrix with log-mel")
    state_matrix = build_state_matrices(
        train_files,
        val_files,
        test_files,
        feature_type="logmel"
    )

    np.savez_compressed("/scratch/almo2783/scratch/audio-reg/state-matrix/state_matrix_logmel.npz", state_matrix)

    print("Building state matrix with MFCC")
    state_matrix = build_state_matrices(
        train_files,
        val_files,
        test_files,
        feature_type="mfcc"
    )

    np.savez_compressed("/scratch/almo2783/scratch/audio-reg/state-matrix/state_matrix_mfcc.npz", state_matrix)