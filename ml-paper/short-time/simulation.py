import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed

@njit(fastmath=True)
def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp):
    # k1
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    # k2
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    # k3
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    # k4
    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    # update y
    for i in range(4):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def rk4_step_inplace_with_force(y, h, c1, c2, c3, c4, c5, phi_dc, a, k1, k2, k3, k4, y_temp, f_x):
    # k1
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy + c5 * f_x
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    # k2
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy + c5 * f_x
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    # k3
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy + c5 * f_x
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    # k4
    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy + c5 * f_x
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    # update y
    for i in range(4):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def simulate_transient(N, h, c1, c2, c3, c4, phi_dc, a):

  y  = np.zeros(4)
  y[0] = 1e-9
  k1 = np.zeros(4)
  k2 = np.zeros(4)
  k3 = np.zeros(4)
  k4 = np.zeros(4)
  y_temp = np.zeros(4)

  for k in range(N):
    rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp)

  return y


@njit(fastmath=True)
def simulate_with_force(y, N, h, c1, c2, c3, c4, c5, phi_dc, a, f_ext):

  k1 = np.zeros(4)
  k2 = np.zeros(4)
  k3 = np.zeros(4)
  k4 = np.zeros(4)
  y_temp = np.zeros(4)
  buf_u_ac = np.empty(N)

  for k in range(N):
    f_x = f_ext[k]
    rk4_step_inplace_with_force(y, h, c1, c2, c3, c4, c5, phi_dc, a, k1, k2, k3, k4, y_temp, f_x)
    buf_u_ac[k] = y[3]

  return buf_u_ac



def build_state_matrix_short_time(
    file_list_path, output_path, a, u_dc, mu,
    frame_len=1024, hop_len=512, n_fft_per_frame=512
):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # === All your constants (same as before) ===
    alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0
    u_max = 1.0
    h = 1e-6 * omega_0
    T = 50.0 * omega_0

    N_trans = int(T / h)
    sample_rate_sim = int((1.0 * omega_0) / h)        # ~16.3 MS/s
    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / (u_max)
    c5 = mu / (l_0 * omega_0**2)
    phi_dc = u_dc / u_max

    # Precompute transient
    y_final = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a)

    # Precompute resampling indices (same for all files assuming fixed length & sr)
    data0, sr = sf.read(filenames[0])
    frac = sample_rate_sim / sr
    total_sim_samples = int(len(data0) * frac)
    idxs = (np.arange(total_sim_samples) / frac).astype(np.int64)
    b_filt, a_filt = butter(4, 0.49, btype='low', fs=1.0)  # now normalize to sim rate

    # Determine number of frames
    n_samples_sim = N_force = total_sim_samples
    frames = []
    for start in range(0, n_samples_sim - frame_len + 1, hop_len):
        frames.append((start, start + frame_len))
    n_frames = len(frames)
    feature_dim = n_fft_per_frame // 2 + 1

    print(f"Using {n_frames} frames per file, {feature_dim} FFT bins per frame")

    def process_file_short_time(fname):
        data, _ = sf.read(fname)
        signal = data[idxs]
        signal = lfilter(b_filt, a_filt, signal - signal.mean())  # remove DC

        y = y_final.copy()
        features = np.zeros((n_frames, feature_dim), dtype=np.float32)

        k1 = np.zeros(4); k2 = np.zeros(4); k3 = np.zeros(4); k4 = np.zeros(4); y_temp = np.zeros(4)

        for i, (start, end) in enumerate(frames):
            u_ac_segment = np.zeros(frame_len, dtype=np.float64)
            seg_len = end - start
            for k in range(seg_len):
                f_x = signal[start + k]
                rk4_step_inplace_with_force(y, h, c1,c2,c3,c4,c5, phi_dc, a,
                                          k1,k2,k3,k4, y_temp, f_x)
                u_ac_segment[k] = y[3]

            # Short FFT of reservoir response in this frame
            fft_mag = np.abs(np.fft.rfft(u_ac_segment, n=n_fft_per_frame))
            features[i] = fft_mag[:feature_dim]

        return features.ravel()  # (n_frames * feature_dim,)

    results = Parallel(n_jobs=64, verbose=1, backend='threading')(delayed(process_file_short_time)(f) for f in filenames)
    state_matrix = np.vstack(results).astype(np.float32)

    np.savez_compressed(output_path,
                        state_matrix=state_matrix,
                        n_frames=n_frames,
                        feature_dim=feature_dim)
    print(f"Saved short-time state matrix: {state_matrix.shape}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        a = float(sys.argv[1])
        u_dc = float(sys.argv[2])
        mu = float(sys.argv[3])
    else:
        # best val point
        a = 0.44
        u_dc = 0.4
        mu = 1.0

    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    build_state_matrix_short_time(
        train_files,
        f'/scratch/almo2783/scratch/ml-paper/short-time/state-matrix/state_matrix_train-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu,
        frame_len=2048, hop_len=512, n_fft_per_frame=1024
    )

    build_state_matrix_short_time(
        val_files,
        f'/scratch/almo2783/scratch/ml-paper/short-time/state-matrix/state_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu,
        frame_len=2048, hop_len=512, n_fft_per_frame=1024
    )

    build_state_matrix_short_time(
        test_files,
        f'/scratch/almo2783/scratch/ml-paper/short-time/state-matrix/state_matrix_test-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu,
        frame_len=2048, hop_len=512, n_fft_per_frame=1024
    )