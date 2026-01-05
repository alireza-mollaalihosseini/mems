import numpy as np
from scipy.integrate import odeint
from numba import njit


@njit(inline='always')
def derivatives (y, t, omega_0, Q_0, beta, alpha, gamma, kappa, u_max, tau, R, phi_dc, a):
  exsi, etta, psy, phi_ac = y

  dot_exsi = etta
  dot_etta = -((omega_0*tau)/Q_0) * etta - (omega_0**2*tau**2) * exsi + (omega_0**2*tau**2) * psy
  temp = a * phi_ac + phi_dc
  min_term = min(temp**2, 1)
  dot_psy = - (beta*tau) * psy + ((alpha * gamma * kappa * tau * u_max) / (omega_0**2 * R**2)) * min_term
  dot_phi_ac = - phi_ac + etta
  return np.array([dot_exsi, dot_etta, dot_psy, dot_phi_ac])

# Parameters
omega_0 = 53956.46373431294
Q_0 = 50.0
alpha = 19.2
beta = 1066.0
gamma = 1.62e7
R = 12.5
tau = 0.001
kappa = 0.602e6
u_dc = 0.7
u_max = 1.0
h = 1e-6
T1 = 50.0
T2 = 500.0
t_rec = 10.0
noise = 1e-3
store_every = 10000  # Store every 10000 steps for plotting

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

# Precompute constants
N1 = int(T1 / h)               # Total number of steps
N2 = int(T2 / h)               # Total number of steps
n_rec = int(t_rec / h)       # Number of steps to record
phi_dc = u_dc / u_max

# Time array for simulation
t1 = np.linspace(0, T1, N1 + 1)
t2 = np.linspace(0, T2, N2 + 1)

# Initialize results array to store the last n_rec values of the first variable
results = np.zeros((len(a_values), n_rec//100))

# Initial conditions: [x, v, theta, u_ac]
y0 = np.array([1e-9, 0.0, 0.0, 0.0])

# Helper to test membership (with float‐tolerance)
def in_extra(a):
    return (np.any(np.isclose(a, extra1, atol=1e-8)) or
            np.any(np.isclose(a, extra2, atol=1e-8)))

# Simulate for each value of a
for i, a in enumerate(a_values[::-1]):

    if (in_extra(a) and a < 0):
        # Use the longer time array for specific 'a' values
        t = t2
        # Solve the ODE
        sol = odeint(derivatives, y0, t, args=(omega_0, Q_0, beta, alpha, gamma, kappa, u_max, tau, R, phi_dc, a))
    else:
        # Use the shorter time array for other 'a' values
        t = t1
        # Solve the ODE
        sol = odeint(derivatives, y0, t, args=(omega_0, Q_0, beta, alpha, gamma, kappa, u_max, tau, R, phi_dc, a))
    
    # Store the last n_rec values of the first variable (e.g., x)
    results[i, :] = sol[-n_rec:, 0][::100]
    
    # Update initial condition for the next iteration
    y0 = sol[-1]
    y0[0] += noise

# Results is now a 2D array with shape (len(a_values), n_rec)
np.save(f'deflections-down-t-sim-{T2}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}.npy', results)