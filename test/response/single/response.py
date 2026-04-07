import numpy as np
from joblib import Parallel, delayed
from numba import njit
import matplotlib.pyplot as plt
plt.style.use("bmh")


@njit(fastmath=True)
def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp):
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def rk4_step_inplace_with_force(y, h, c1, c2, c3, c4, c5, phi_dc, a, k1, k2, k3, k4, y_temp, f_x):
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy + c5 * f_x
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy + c5 * f_x
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy + c5 * f_x
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy + c5 * f_x
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

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


def generate_chirp(
        fs,
        duration,
        f_start,
        f_end,
        amplitude=1.0,
        method="log"
    ):
    """
    Generate chirp signal.

    fs : sample rate
    duration : seconds
    f_start, f_end : Hz
    amplitude : signal amplitude
    """

    t = np.arange(0, duration, 1/fs)

    if method == "linear":
        k = (f_end - f_start) / duration
        phase = 2*np.pi*(f_start*t + 0.5*k*t**2)

    elif method == "log":
        k = np.log(f_end/f_start) / duration
        phase = 2*np.pi * f_start * (np.exp(k*t)-1)/k

    signal = amplitude * np.sin(phase)

    return signal.astype(np.float32)


def compute_peak_gain(input_signal, output_signal, fs):

    X = np.fft.rfft(input_signal)
    Y = np.fft.rfft(output_signal)

    H = Y / (X + 1e-12)

    gain = 20*np.log10(np.abs(H))

    peak_gain = np.max(gain)
    peak_loc = np.argmax(gain)

    freqs = np.fft.rfftfreq(len(input_signal), 1/fs)
    f_res = freqs[peak_loc]

    return peak_gain, f_res


def run_simulation(mu, Q_0, l_0, omega_0, h, c1, c3, c4, phi_dc, a):

    c2 = 1 / Q_0
    c5 = mu / (l_0 * omega_0**2)
    fs = 1e6
    duration = 1.0          # seconds
    new_len = int(fs * duration)

    signal = generate_chirp(
        fs,
        duration,
        f_start=10,
        f_end=20_000,
        amplitude=mu
    )

    # transient
    y_final = simulate_transient(
        50_000_000,
        h,
        c1, c2, c3, c4,
        phi_dc,
        a
    )

    # forced response
    u_ac_buf = simulate_with_force(
        y_final,
        new_len,
        h,
        c1, c2, c3, c4, c5,
        phi_dc,
        a,
        signal
    )

    peak_gain, f_res = compute_peak_gain(signal, u_ac_buf, fs)

    return peak_gain, f_res




f = 1000
a = 0.3
u_dc = 0.1
# mu = 1.0
# mu_values = np.linspace(1.0, 100.0, 100, dtype=int)
mu_values = np.logspace(-2, 2, 501)
Q_values = np.array([20.0, 50.0, 100.0, 200.0, 500.0])

omega_0 = f * 2 * np.pi
h = 1e-6 * omega_0

alpha, tau, beta, gamma, R, kappa = 19.2, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
u_max = 1.0
l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
c1 = beta / omega_0
# c2 = 1 / Q_0
c3 = 1 / (tau * omega_0)
c4 = (kappa * l_0) / u_max
phi_dc = u_dc / u_max



results = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
    delayed(run_simulation)(mu, Q, l_0, omega_0, h, c1, c3, c4, phi_dc, a)
    for mu in mu_values
    for Q in Q_values
)


peak_gain_map = np.zeros((len(mu_values), len(Q_values)))
f_res_map = np.zeros_like(peak_gain_map)

k = 0
for i in range(len(mu_values)):
    for j in range(len(Q_values)):
        peak_gain_map[i, j], f_res_map[i, j] = results[k]
        k += 1


np.save("peak_gain_map.npy", peak_gain_map)
np.save("resonance_shift_map.npy", f_res_map)


plt.figure(figsize=(7,5))

plt.imshow(
    peak_gain_map,
    origin="lower",
    aspect="auto",
    extent=[Q_values[0], Q_values[-1],
            mu_values[0], mu_values[-1]]
)

plt.colorbar(label="Peak Gain (dB)")
plt.yscale("log")
plt.xlabel("Q factor")
plt.ylabel("Amplitude μ")
plt.title("Sensor Sensitivity Map")
# plt.show()
plt.savefig("peak_gain_map.png", dpi=300)
plt.tight_layout()
plt.close()


plt.figure(figsize=(7,5))

plt.imshow(
    f_res_map,
    origin="lower",
    aspect="auto",
    extent=[Q_values[0], Q_values[-1],
            mu_values[0], mu_values[-1]]
)

plt.colorbar(label="Resonance Frequency (Hz)")
plt.yscale("log")
plt.xlabel("Q factor")
plt.ylabel("Amplitude μ")
plt.title("Resonance Shift")
# plt.show()
plt.savefig("resonance_shift_map.png", dpi=300)
plt.close()