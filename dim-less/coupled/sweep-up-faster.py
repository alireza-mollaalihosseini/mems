import sys
import numpy as np
from numba import njit

@njit(fastmath=True)
def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, cf, a, k1, k2, k3, k4, y_temp):
    # k1
    x1, nu1, th1, u1, x2, nu2, th2, u2 = y
    temp1 = a * u1 + cf * u2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a * u2 + cf * u1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k1[0] = nu1
    k1[1] = -c2 * nu1 - x1 + th1
    k1[2] = -c1 * th1 + c1 * min_term1
    k1[3] = -c3 * u1 + c4 * nu1
    k1[4] = nu2
    k1[5] = -c2 * nu2 - x2 + th2
    k1[6] = -c1 * th2 + c1 * min_term2
    k1[7] = -c3 * u2 + c4 * nu2

    # k2
    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    x1, nu1, th1, u1, x2, nu2, th2, u2 = y_temp
    temp1 = a * u1 + cf * u2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a * u2 + cf * u1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k2[0] = nu1
    k2[1] = -c2 * nu1 - x1 + th1
    k2[2] = -c1 * th1 + c1 * min_term1
    k2[3] = -c3 * u1 + c4 * nu1
    k2[4] = nu2
    k2[5] = -c2 * nu2 - x2 + th2
    k2[6] = -c1 * th2 + c1 * min_term2
    k2[7] = -c3 * u2 + c4 * nu2

    # k3
    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    x1, nu1, th1, u1, x2, nu2, th2, u2 = y_temp
    temp1 = a * u1 + cf * u2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a * u2 + cf * u1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k3[0] = nu1
    k3[1] = -c2 * nu1 - x1 + th1
    k3[2] = -c1 * th1 + c1 * min_term1
    k3[3] = -c3 * u1 + c4 * nu1
    k3[4] = nu2
    k3[5] = -c2 * nu2 - x2 + th2
    k3[6] = -c1 * th2 + c1 * min_term2
    k3[7] = -c3 * u2 + c4 * nu2

    # k4
    for i in range(8):
        y_temp[i] = y[i] + h * k3[i]
    x1, nu1, th1, u1, x2, nu2, th2, u2 = y_temp
    temp1 = a * u1 + cf * u2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a * u2 + cf * u1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k4[0] = nu1
    k4[1] = -c2 * nu1 - x1 + th1
    k4[2] = -c1 * th1 + c1 * min_term1
    k4[3] = -c3 * u1 + c4 * nu1
    k4[4] = nu2
    k4[5] = -c2 * nu2 - x2 + th2
    k4[6] = -c1 * th2 + c1 * min_term2
    k4[7] = -c3 * u2 + c4 * nu2

    # update y
    for i in range(8):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def run_simulation(a_values, N, n_rec, h, c1, c2, c3, c4, phi_dc, cf, noise):
    results = np.zeros((len(a_values), 2, 50000))

    y = np.zeros(8)
    y[0] = 1e-9
    y[4] = 1e-9

    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)
    # buf_x = np.empty(n_rec)

    for i in range(len(a_values)):
        a = a_values[i]
        for k in range(N):
            rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, cf, a, k1, k2, k3, k4, y_temp)
            if k >= N - 50000:
                idx = k - (N - 50000)
                results[i, 0, idx] = y[0]
                results[i, 1, idx] = y[4]
        y[0] += noise
        y[4] += noise

    return results


# === parameters ===
omega_0 = 53956.46373431294
Q_0     = 50.0
alpha   = 19.2
beta    = 1066.0
gamma   = 1.62e7
R       = 12.5
tau     = 0.001
kappa   = 0.602e6
u_dc    = 0.7
u_max   = 1.0
# c_f     = 0.1
h       = omega_0 * 1e-6
T       = omega_0 * 500.0
t_rec   = omega_0 * 3.0
l_0     = alpha * gamma * u_max**2 / (beta * R**2 * omega_0**2)
noise   = 1e-10 / l_0

a_values = np.linspace(-3, 4, 1001)

c_f = float(sys.argv[1])

N            = int(T / h)
n_rec        = int(t_rec / h)
c1           = beta / omega_0
c2           = 1/ Q_0
c3           = 1 / (tau * omega_0)
c4           = (alpha * gamma * kappa * u_max) / (beta * R**2 * omega_0**2)
phi_dc       = u_dc / u_max

# Run simulation
results = run_simulation(a_values, N, n_rec, h, c1, c2, c3, c4, phi_dc, c_f, noise)

np.save(
    f'/scratch/almo2783/scratch/dim-less/coupled/deflections-up/deflections-RK4-up-t-sim-{int(T / omega_0)}-t-rec-{int(t_rec / omega_0)}-noise-{noise:.0e}-cf-{c_f}.npy',
    results
)