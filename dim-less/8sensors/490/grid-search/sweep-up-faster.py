import sys
import numpy as np
from numba import njit

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
def run_simulation(a_values, N, n_rec, h, c1, c2, c3, c4, phi_dc, noise):
    # results = np.zeros((len(a_values), n_rec // 100))
    results = np.zeros((len(a_values), 500000))
    y = np.zeros(4)
    y[0] = 1e-9

    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    y_temp = np.zeros(4)
    # buf_x = np.empty(n_rec)
    # buf_x = np.empty(500000)

    for i in range(len(a_values)):
        a = a_values[i]
        for k in range(N):
            rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp)
            if k >= N - 500000:
                # buf_x[k - (N - n_rec)] = y[0]
                # buf_x[k - (N - n_rec)] = y[3]
                # buf_x[k - (N - 500000)] = y[0]
                results[i, k - (N - 500000)] = y[0]
        # results[i, :] = buf_x
        y[0] += noise
    return results


# === parameters ===
# Simulation params
alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 490.0, 3078.7608005179973, 33.0, 0.0039000000000000003, 256.4102564102564, 1800000.0, 13.4, 162540.0
u_max = 1.0
h = 1e-6 * omega_0
T = 50.0 * omega_0
t_rec   = omega_0 * 3.0
l_0     = alpha * gamma * u_max**2 / (beta * R**2 * omega_0**2)
noise   = 1e-10 / l_0

a_values = np.linspace(-1, 1, 101)

u_dc = float(sys.argv[1])

N            = int(T / h)
n_rec        = int(t_rec / h)
c1           = beta / omega_0
c2           = 1/ Q_0
c3           = 1 / (tau * omega_0)
c4           = (alpha * gamma * kappa * u_max) / (beta * R**2 * omega_0**2)
phi_dc       = u_dc / u_max

# Run simulation
results = run_simulation(a_values, N, n_rec, h, c1, c2, c3, c4, phi_dc, noise)

np.save(
    f'/scratch/almo2783/scratch/dim-less/8sensors/490/grid-search/deflections-up/deflections-RK4-up-t-sim-{int(T / omega_0)}-t-rec-{int(t_rec / omega_0)}-noise-{noise:.0e}-u_max-{u_max}-u_dc-{u_dc}.npy',
    results
)