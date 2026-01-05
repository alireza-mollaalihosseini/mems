import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfiltfilt
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


# def build_state_matrix_spectral_mixer(
#     file_list_path, output_path, a, u_dc, mu,
#     n_delays=8, max_delay_ms=8.0
# ):
#     filenames = np.loadtxt(file_list_path, dtype=str)

#     # === Same constants and setup ===
#     alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0
#     u_max = 1.0
#     h = 1e-6 * omega_0
#     T = 50.0 * omega_0
#     N_trans = int(T / h)
#     sample_rate_sim = int(omega_0 / h)
#     l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
#     c1,c2,c3,c4,c5 = beta/omega_0, 1/Q_0, 1/(tau*omega_0), (kappa*l_0)/u_max, mu/(l_0*omega_0**2)
#     phi_dc = u_dc / u_max

#     y_final = simulate_transient(N_trans, h, c1,c2,c3,c4, phi_dc, a)

#     data0, sr = sf.read(filenames[0])
#     frac = sample_rate_sim / sr
#     idxs = (np.arange(int(len(data0)*frac)) / frac).astype(np.int64)
#     b_filt, a_filt = butter(4, 0.49, btype='low', fs=1.0)

#     # Delay taps in samples
#     max_delay_samples = int(max_delay_ms * 0.001 * sample_rate_sim)
#     delay_taps = np.linspace(0, max_delay_samples, n_delays, endpoint=False).astype(int)

#     fft_len_per_channel = 8001
#     total_features = fft_len_per_channel * n_delays

#     def process_file_mixer(fname):
#         data, _ = sf.read(fname)
#         signal = lfilter(b_filt, a_filt, data[idxs])

#         features = np.zeros((n_delays, fft_len_per_channel), dtype=np.float32)
#         y = y_final.copy()
#         k1=k2=k3=k4=np.zeros(4); y_temp=np.zeros(4)

#         # We run ONE long simulation, but inject delayed versions additively
#         delayed_signals = np.zeros(n_delays)
#         for k in range(len(signal)):
#             current_sample = signal[k]

#             # Update delay line (circular)
#             for d in range(n_delays):
#                 delay_idx = (k - delay_taps[d]) % len(signal)
#                 if delay_idx >= 0:
#                     delayed_signals[d] = signal[delay_idx]
#                 else:
#                     delayed_signals[d] = 0.0

#             # Inject sum of delayed + original (virtual multi-sensor)
#             f_x = current_sample + 0.3 * delayed_signals.sum()  # mixing gain

#             rk4_step_inplace_with_force(y, h, c1,c2,c3,c4,c5, phi_dc, a,
#                                       k1,k2,k3,k4, y_temp, f_x)

#             if k % 100 == 0:  # collect reservoir output sparsely
#                 features[0, k//100] = y[3]  # just store time series for now

#         # After full run, take FFT of the mixed reservoir response
#         fft_mag = np.abs(np.fft.rfft(y[3] * np.hanning(len(signal))[:len(signal)//100*100]))
#         return np.tile(fft_mag[:fft_len_per_channel], n_delays)[:total_features]  # placeholder

#         # BETTER: run n_delays separate simulations, each driven by one delayed version
#         # (more orthogonal, usually better)

#     # === RECOMMENDED: parallel delayed drives (best performance) ===
#     def process_file_multi_delay(fname):
#         data, _ = sf.read(fname)
#         signal = lfilter(b_filt, a_filt, data[idxs].astype(np.float64))
#         feats = []

#         for delay_samples in delay_taps:
#             y = y_final.copy()
#             delayed_sig = np.roll(signal, delay_samples)
#             if delay_samples > 0:
#                 delayed_sig[:delay_samples] = 0

#             u_ac = simulate_with_force(y, len(signal), h, c1,c2,c3,c4,c5,
#                                      phi_dc, a, delayed_sig)
#             fft_mag = np.abs(np.fft.rfft(u_ac))[:(fft_len_per_channel)]
#             feats.append(fft_mag)

#         return np.concatenate(feats).astype(np.float32)

#     results = Parallel(n_jobs=64, verbose=1, backend='threading')(delayed(process_file_multi_delay)(f) for f in filenames)
#     state_matrix = np.vstack(results)

#     np.savez_compressed(output_path, state_matrix=state_matrix)
#     print(f"Spectral mixer matrix: {state_matrix.shape}")


def build_state_matrix_multi_band(
    file_list_path, output_path, a, u_dc, mu,
    n_bands=8, fft_len_per_band=8001
):
    """
    Physical reservoir as multi-band nonlinear compressor.
    Best version for TAU Urban Acoustic Scenes 2022 (Barcelona + Device A).
    """
    filenames = np.loadtxt(file_list_path, dtype=str)
    print(f"Processing {len(filenames)} files → multi-band reservoir (a={a}, u_dc={u_dc}, mu={mu})")

    # === Physical system constants (same as yours) ===
    alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = (
        12.5, 2600.0, 16336.281798666923, 82.0, 0.0091,
        109.89010989010988, 26827200.0, 50.24, 884700.0
    )
    u_max = 1.0
    h = 1e-6 * omega_0
    T = 50.0 * omega_0
    N_trans = int(T / h)
    sample_rate_sim = int(omega_0 / h)          # ~16.3 MS/s

    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / u_max
    c5 = mu / (l_0 * omega_0**2)
    phi_dc = u_dc / u_max

    # Transient washout
    y_init = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a)

    # Resampling setup (44.1 kHz → sim rate)
    data0, sr = sf.read(filenames[0])
    assert sr == 44100 or sr == 48000, "Expected 44.1 or 48 kHz"
    frac = sample_rate_sim / sr
    total_sim_samples = int(len(data0) * frac)
    idxs = (np.arange(total_sim_samples) / frac).astype(np.int64)

    # Anti-aliasing lowpass (normalized to sim rate)
    b_filt, a_filt = butter(4, 0.49, btype='low', fs=1.0)

    # === 8 Urban-tuned frequency bands (Hz) ===
    band_edges = [0, 125, 250, 500, 1000, 2000, 4000, 8000, 22050]  # Nyquist ~22.05k
    sos_bands = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        if high == 22050:
            sos = butter(4, low / (sr / 2), btype='high', output='sos')
        elif low == 0:
            sos = butter(4, high / (sr / 2), btype='low', output='sos')
        else:
            sos = butter(4, [low / (sr / 2), high / (sr / 2)], btype='band', output='sos')
        sos_bands.append(sos)

    total_features = fft_len_per_band * n_bands

    def process_file(fname):
        data, _ = sf.read(fname)
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono
        signal = data.astype(np.float64)

        # Resample to simulation rate
        signal_sim = np.zeros(total_sim_samples, dtype=np.float64)
        signal_sim[:len(signal)] = signal
        signal_sim = lfilter(b_filt, a_filt, signal_sim)  # anti-alias
        signal_sim = signal_sim[idxs]

        # Normalize per clip (critical!)
        if signal_sim.std() > 0:
            signal_sim = signal_sim / (2.5 * signal_sim.std())

        features = np.zeros(total_features, dtype=np.float32)
        pos = 0

        for i, sos in enumerate(sos_bands):
            band_sig = sosfiltfilt(sos, signal_sim)  # zero-phase filtering

            y = y_init.copy()
            u_ac = simulate_with_force(
                y, total_sim_samples, h,
                c1, c2, c3, c4, c5, phi_dc, a,
                band_sig * 0.8  # slight per-band gain to avoid saturation
            )

            fft_mag = np.abs(np.fft.rfft(u_ac, n=fft_len_per_band * 2 - 1))
            features[pos:pos + fft_len_per_band] = fft_mag[:fft_len_per_band]
            pos += fft_len_per_band

        return features

    # === Parallel processing ===
    results = Parallel(n_jobs=64, verbose=1, backend='threading')(
        delayed(process_file)(fname) for fname in filenames
    )

    state_matrix = np.vstack(results)
    print(f"Multi-band state matrix shape: {state_matrix.shape} (features = {total_features})")

    np.savez_compressed(output_path, state_matrix=state_matrix)
    print(f"Saved to {output_path}")


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

    build_state_matrix_multi_band(
        train_files,
        f'/scratch/almo2783/scratch/ml-paper/spectral-mixer/state-matrix/state_matrix_train-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu,
        n_bands=8, fft_len_per_band=8001
    )

    build_state_matrix_multi_band(
        val_files,
        f'/scratch/almo2783/scratch/ml-paper/spectral-mixer/state-matrix/state_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu,
        n_bands=8, fft_len_per_band=8001
    )

    build_state_matrix_multi_band(
        test_files,
        f'/scratch/almo2783/scratch/ml-paper/spectral-mixer/state-matrix/state_matrix_test-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu,
        n_bands=8, fft_len_per_band=8001
    )