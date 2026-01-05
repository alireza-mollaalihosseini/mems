import numpy as np
from numba import njit

# --------------------
# Derivative function
# --------------------
@njit(inline='always')
def derivatives(y, t, phi_dc, a, c1, c2, c3, c4):
    exsi, etta, psy, phi_ac = y

    dot_exsi = etta
    dot_etta = -c1 * etta - c2 * exsi + c2 * psy
    temp = a * phi_ac + phi_dc
    min_term = min(temp**2, 1.0)
    dot_psy = -c3 * psy + c3 * min_term
    dot_phi_ac = -phi_ac + c4 * etta

    return np.array([dot_exsi, dot_etta, dot_psy, dot_phi_ac])

# --------------------
# RK4 integrator
# --------------------
def rk4_step(y, t, h, phi_dc, a, c1, c2, c3, c4):
    k1 = derivatives(y, t, phi_dc, a, c1, c2, c3, c4)
    k2 = derivatives(y + 0.5*h*k1, t + 0.5*h, phi_dc, a, c1, c2, c3, c4)
    k3 = derivatives(y + 0.5*h*k2, t + 0.5*h, phi_dc, a, c1, c2, c3, c4)
    k4 = derivatives(y + h*k3, t + h, phi_dc, a, c1, c2, c3, c4)

    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(y0, t_max, h, phi_dc, a, c1, c2, c3, c4):
    n_steps = int(t_max/h)
    y = np.zeros((n_steps, len(y0)))
    t = np.zeros(n_steps)
    y[0] = y0

    for i in range(1, n_steps):
        t[i] = t[i-1] + h
        y[i] = rk4_step(y[i-1], t[i-1], h, phi_dc, a, c1, c2, c3, c4)

    return t, y

# --------------------
# Parameters
# --------------------
omega_0 = 53956.46373431294
Q_0     = 50.0
alpha   = 19.2
beta    = 1066
gamma   = 1.62e7
R       = 12.5
tau     = 0.001
kappa   = 0.602e6
u_dc    = 0.7
u_max   = 1.0
kick    = 1e-3

c1           = omega_0 * tau / Q_0
c2           = omega_0**2 * tau**2
c3           = beta * tau
c4           = (alpha * gamma * kappa * u_max) / (beta * R**2 * omega_0**2)
phi_dc       = u_dc / u_max
# a = -0.5
# a_values = np.linspace(-2, 4, 1001)

# Original ranges
a1 = np.arange(-2.0, -1.9, 0.1)
a2 = np.arange(-1.9, -1.8, 0.0005)
a3 = np.arange(-1.8, 3.5, 0.1)
a4 = np.arange(3.5, 3.7, 0.0005)
a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# Additional 20 points between specified intervals
extra1 = np.linspace(-1.8775, -1.875, 200)
extra2 = np.linspace(3.5825, 3.585, 200)

# Combine all and sort
a_values = np.concatenate([a1, a2, extra1, a3, a4, extra2, a5])
a_values = np.unique(np.round(a_values, decimals=8))

# Initial condition
y0 = np.array([1e-9, 0.0, 0.0, 0.0])
h = 1e-3
t_max = 6000
# deflections = np.empty(10000)
# results = np.zeros((len(a_values), 10000)

# Arrays for storing results
max_peaks = np.zeros(len(a_values))
min_peaks = np.zeros(len(a_values))

for i, a in enumerate(a_values[::-1]):
    t, y = simulate(y0, t_max, h, phi_dc, a, c1, c2, c3, c4)
    # results[i, :] = y[:, 0][-10000:]
    max_peaks[i] = np.max(y[-1000:, 0])
    min_peaks[i] = np.min(y[-1000:, 0])
    y0 = y[-1]
    y0[0] += kick


np.save(f'max_peaks-down-tau-kick-{kick}.npy', max_peaks)
np.save(f'min_peaks-down-tau-kick-{kick}.npy', min_peaks)
