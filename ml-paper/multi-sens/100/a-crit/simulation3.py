import numpy as np
from numba import njit
from joblib import Parallel, delayed

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
def simulate_and_measure_amp(N, h, c1, c2, c3, c4, phi_dc, a):
    y = np.zeros(4)
    y[0] = 1e-9

    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    y_temp = np.zeros(4)

    u_min = 1e30
    u_max = -1e30

    for _ in range(N):
        rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a,
                         k1, k2, k3, k4, y_temp)
        u = y[3]
        if u < u_min:
            u_min = u
        if u > u_max:
            u_max = u

    return u_max - u_min



def find_positive_hopf(
    a_vals, f, N, alpha, Q_0, tau, beta, gamma, R, kappa, u_dc, u_max,
    jump_factor=5.0
):

    omega_0 = f * 2 * np.pi
    h = 1e-6 * omega_0

    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)

    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / u_max
    phi_dc = u_dc / u_max
    
    pre_amp = 0.0
    pre_mean = 1e30
    for a in a_vals:
        amp = simulate_and_measure_amp(
            N, h, c1, c2, c3, c4, phi_dc, a
        )
        if np.mean(amp+pre_amp) > jump_factor * pre_mean:
            return a
        
        pre_mean = np.mean(amp + pre_amp)
        pre_amp  = amp

    return np.nan


def calculate_a_crit(u_dc, a_values, f_values):

    # a_list = []
    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0
    N = 100_000_000

    a_crit = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(find_positive_hopf)(
            a_values, f, N, alpha, Q_0, tau, beta, gamma, R, kappa, u_dc, u_max
        ) for f in f_values
    )

    return np.array(a_crit)


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--u_dc', type=float, required=True, help='Value of u_dc to process')
args = parser.parse_args()
u_dc = args.u_dc
# # u_dc_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# u_dc_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# u_dc_values = np.array([0.1])

a_values = np.arange(0, 2000, 0.1)
f_values = np.linspace(1000, 50000, 100, dtype=int)

# for u_dc in u_dc_values:
print(f"\nProcessing u_dc = {u_dc:.1f}")
a_vals = calculate_a_crit(u_dc, a_values, f_values)
np.save(f"a-crit-u-dc-{u_dc:.1f}-more-a.npy", a_vals)