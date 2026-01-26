import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

@njit(fastmath=True)
def rk4_step(y, h, c1, c2, c3, c4, phi_dc, a, c5_fx, k1, k2, k3, k4, y_temp):
    # k1
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy + c5_fx
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    # k2
    y_temp[:] = y + 0.5 * h * k1
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy + c5_fx
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    # k3
    y_temp[:] = y + 0.5 * h * k2
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy + c5_fx
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    # k4
    y_temp[:] = y + h * k3
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy + c5_fx
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    # update
    y += (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@njit(fastmath=True)
def simulate_transient(N, h, c1, c2, c3, c4, phi_dc, a):
    y = np.zeros(4)
    y[0] = 1e-9
    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    y_temp = np.zeros(4)

    for _ in range(N):
        rk4_step(y, h, c1, c2, c3, c4, phi_dc, a, 0.0, k1, k2, k3, k4, y_temp)

    return y.copy()


@njit(fastmath=True)
def simulate_with_force(y_start, N, h, c1, c2, c3, c4, c5, phi_dc, a, f_ext):
    y = y_start.copy()
    buf_u_ac = np.empty(N, dtype=np.float64)
    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    y_temp = np.zeros(4)

    for k in range(N):
        c5_fx = c5 * f_ext[k]
        rk4_step(y, h, c1, c2, c3, c4, phi_dc, a, c5_fx, k1, k2, k3, k4, y_temp)
        buf_u_ac[k] = y[3]

    return buf_u_ac


def compute_transient(f, a, mu, u_dc):
    omega_0 = f * 2 * np.pi
    h = 1e-6 * omega_0

    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0
    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / u_max
    c5 = mu / (l_0 * omega_0**2)
    phi_dc = u_dc / u_max

    y_final = simulate_transient(50000000, h, c1, c2, c3, c4, phi_dc, a)
    return (y_final, h, c1, c2, c3, c4, c5, phi_dc)



if __name__ == '__main__':
    a = 0.9
    mu = 1.0 
    u_dc = 1.0
    
    f_values = np.sort(np.array([43630, 44120, 42650, 45590,  6390, 44610, 23540, 45100,  6880,
       42160,  2960, 46080, 43140, 24030, 49510, 39220,  3940,  5410,
       38730, 37260,  3450, 25010, 20600, 20110,  5900, 41180,  4430,
        4920, 24520, 47060, 40200, 37750, 41670, 46570, 50000, 39710,
       38240, 36770, 48040, 48530,  7370, 18640, 49020,  2470, 47550,
       40690, 22070, 36280, 23050,  7860, 19620, 18150, 19130, 35300,
       21090, 28930, 21580, 25990,  8840, 11290,  8350,  9330,  1980,
       35790]))
    
    # Parallel precomputation of transients
    precomp_list = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(compute_transient)(f, a, mu, u_dc) for f in f_values
    )

    precomp_list = np.array(precomp_list, dtype=object)
    
    # Save precomputed parameters to file
    np.save("precomputed_params.npy", precomp_list)