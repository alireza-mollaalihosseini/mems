import numpy as np
from numba import njit
from numba import prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


@njit(fastmath=True)
def rk4_step_inplace_coupled(y, h,
                             c1_1, c2_1, c3_1, c4_1,
                             c1_2, c2_2, c3_2, c4_2,
                             phi_dc, a1, a2, g,
                             k1, k2, k3, k4, y_temp):
    exsi1 = y[0]; etta1 = y[1]; psy1 = y[2]; phi1 = y[3]
    exsi2 = y[4]; etta2 = y[5]; psy2 = y[6]; phi2 = y[7]

    # k1
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k1[0] = etta1
    k1[1] = -c2_1 * etta1 - exsi1 + psy1
    k1[2] = -c1_1 * psy1 + c1_1 * min_term1
    k1[3] = -c3_1 * phi1 + c4_1 * etta1
    k1[4] = etta2
    k1[5] = -c2_2 * etta2 - exsi2 + psy2
    k1[6] = -c1_2 * psy2 + c1_2 * min_term2
    k1[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k2
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k2[0] = etta1
    k2[1] = -c2_1 * etta1 - exsi1 + psy1
    k2[2] = -c1_1 * psy1 + c1_1 * min_term1
    k2[3] = -c3_1 * phi1 + c4_1 * etta1
    k2[4] = etta2
    k2[5] = -c2_2 * etta2 - exsi2 + psy2
    k2[6] = -c1_2 * psy2 + c1_2 * min_term2
    k2[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k3
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k3[0] = etta1
    k3[1] = -c2_1 * etta1 - exsi1 + psy1
    k3[2] = -c1_1 * psy1 + c1_1 * min_term1
    k3[3] = -c3_1 * phi1 + c4_1 * etta1
    k3[4] = etta2
    k3[5] = -c2_2 * etta2 - exsi2 + psy2
    k3[6] = -c1_2 * psy2 + c1_2 * min_term2
    k3[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + h * k3[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k4
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k4[0] = etta1
    k4[1] = -c2_1 * etta1 - exsi1 + psy1
    k4[2] = -c1_1 * psy1 + c1_1 * min_term1
    k4[3] = -c3_1 * phi1 + c4_1 * etta1
    k4[4] = etta2
    k4[5] = -c2_2 * etta2 - exsi2 + psy2
    k4[6] = -c1_2 * psy2 + c1_2 * min_term2
    k4[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def rk4_step_inplace_with_force_coupled(y, h,
                                        c1_1, c2_1, c3_1, c4_1, c5_1,
                                        c1_2, c2_2, c3_2, c4_2, c5_2,
                                        phi_dc, a1, a2, g,
                                        k1, k2, k3, k4, y_temp, f_x):
    exsi1 = y[0]; etta1 = y[1]; psy1 = y[2]; phi1 = y[3]
    exsi2 = y[4]; etta2 = y[5]; psy2 = y[6]; phi2 = y[7]

    # k1
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k1[0] = etta1
    k1[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k1[2] = -c1_1 * psy1 + c1_1 * min_term1
    k1[3] = -c3_1 * phi1 + c4_1 * etta1
    k1[4] = etta2
    k1[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k1[6] = -c1_2 * psy2 + c1_2 * min_term2
    k1[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k2
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k2[0] = etta1
    k2[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k2[2] = -c1_1 * psy1 + c1_1 * min_term1
    k2[3] = -c3_1 * phi1 + c4_1 * etta1
    k2[4] = etta2
    k2[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k2[6] = -c1_2 * psy2 + c1_2 * min_term2
    k2[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k3
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k3[0] = etta1
    k3[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k3[2] = -c1_1 * psy1 + c1_1 * min_term1
    k3[3] = -c3_1 * phi1 + c4_1 * etta1
    k3[4] = etta2
    k3[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k3[6] = -c1_2 * psy2 + c1_2 * min_term2
    k3[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + h * k3[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k4
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 + g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k4[0] = etta1
    k4[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k4[2] = -c1_1 * psy1 + c1_1 * min_term1
    k4[3] = -c3_1 * phi1 + c4_1 * etta1
    k4[4] = etta2
    k4[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k4[6] = -c1_2 * psy2 + c1_2 * min_term2
    k4[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def simulate_transient_coupled(N, h,
                               c1_1, c2_1, c3_1, c4_1,
                               c1_2, c2_2, c3_2, c4_2,
                               phi_dc, a1, a2, g):
    y = np.zeros(8)
    y[0] = 1e-9
    y[4] = 1e-9
    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)

    for k in range(N):
        rk4_step_inplace_coupled(y, h,
                                 c1_1, c2_1, c3_1, c4_1,
                                 c1_2, c2_2, c3_2, c4_2,
                                 phi_dc, a1, -a2, g,
                                 k1, k2, k3, k4, y_temp)
    return y


@njit(fastmath=True)
def simulate_with_force_coupled(y, N, h,
                                c1_1, c2_1, c3_1, c4_1, c5_1,
                                c1_2, c2_2, c3_2, c4_2, c5_2,
                                phi_dc, a1, a2, g, f_ext):
    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)
    buf_x1 = np.empty(N)
    buf_x2 = np.empty(N)

    for k in range(N):
        f_x = f_ext[k]
        rk4_step_inplace_with_force_coupled(y, h,
                                            c1_1, c2_1, c3_1, c4_1, c5_1,
                                            c1_2, c2_2, c3_2, c4_2, c5_2,
                                            phi_dc, a1, -a2, g,
                                            k1, k2, k3, k4, y_temp, f_x)
        buf_x1[k] = y[0]
        buf_x2[k] = y[4]

    return buf_x1, buf_x2


@njit(fastmath=True)
def get_unique_peaks_count(data, tol=1e-4):
    """Fast peak finding and unique counting in Numba.
    
    Used for extrema detection (local maxima) in steady-state sensor output
    to identify the number of distinct coexisting attractors / oscillation levels
    in the coupled system. No change needed from the original for the coupled case.
    """
    peaks = []
    # Identify local maxima (extrema of interest for these multistable systems)
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            val = round(data[i] / tol) * tol
            # Check uniqueness manually (Numba list friendly)
            unique = True
            for p in peaks:
                if abs(p - val) < 1e-6:
                    unique = False
                    break
            if unique:
                peaks.append(val)
    
    if len(peaks) == 0: return 0
    
    # Standard deviation check for unique maxima (stability check)
    p_arr = np.array(peaks)
    if np.std(p_arr) < 0.01:
        return 1
    return len(peaks)


@njit(parallel=True, fastmath=True)
def run_simulation_parallel(a_values, g_values, u_dc_values, N, h,
                            c1_1, c2_1, c3_1, c4_1,
                            c1_2, c2_2, c3_2, c4_2):
    """Main parallel loop for coupled sensors grid search over a, g, and u_dc
    (no additional f_ext / external force).
    
    a is applied symmetrically as a1 = a2 = a (standard for symmetric coupled sensor studies).
    Returns a 3D array of unique peak counts (number of distinct steady-state extrema levels)
    with shape (len(a_values), len(g_values), len(u_dc_values)).
    Uses the provided rk4_step_inplace_coupled for the 8D coupled system.
    Records only the last record_size points of sensor 1 output (y[0]) for memory efficiency.
    """
    n_a = len(a_values)
    n_g = len(g_values)
    n_u = len(u_dc_values)
    results = np.zeros((n_a, n_g, n_u), dtype=np.int32)
    
    total = n_a * n_g * n_u
    record_size = 500_000
    
    for flat_idx in prange(total):
        # Decode 3D indices from flat parallel index
        i = flat_idx // (n_g * n_u)
        j = (flat_idx // n_u) % n_g
        k = flat_idx % n_u
        
        a = a_values[i]      # a1 = a2 = a (symmetric)
        g = g_values[j]
        u_dc = u_dc_values[k]
        phi_dc = u_dc / 1.0  # u_max = 1.0
        
        # Coupled 8D state: [exsi1, etta1, psy1, phi1, exsi2, etta2, psy2, phi2]
        y = np.zeros(8)
        y[0] = 1e-9
        y[4] = 1e-9
        
        # Only record what we need to save memory
        recorded_data = np.zeros(record_size)
        
        # RK4 workspace (allocated per thread)
        k1 = np.zeros(8)
        k2 = np.zeros(8)
        k3 = np.zeros(8)
        k4 = np.zeros(8)
        y_temp = np.zeros(8)
        
        for step in range(N):
            rk4_step_inplace_coupled(y, h,
                                     c1_1, c2_1, c3_1, c4_1,
                                     c1_2, c2_2, c3_2, c4_2,
                                     phi_dc, a, a, g,          # a1 = a, a2 = a
                                     k1, k2, k3, k4, y_temp)
            
            # Record only steady-state tail (same strategy as original)
            if step >= (N - record_size):
                recorded_data[step - (N - record_size)] = y[0]  # sensor 1 output
        
        # Stats and peak counting (same logic as original)
        if np.std(recorded_data) < 1e-8:
            results[i, j, k] = 0
        else:
            results[i, j, k] = get_unique_peaks_count(recorded_data)
            
    return results


# =============================================================================
# Main execution block
# =============================================================================
if __name__ == "__main__":

    f1 = 1000
    f2 = 1494

    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0

    # Sensor 1
    omega_0_1 = f1 * 2 * np.pi
    h1 = 1e-6 * omega_0_1
    l_0_1 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0_1**2)
    c1_1 = beta / omega_0_1
    c2_1 = 1 / Q_0
    c3_1 = 1 / (tau * omega_0_1)
    c4_1 = (kappa * l_0_1) / u_max
    # c5_1 = mu / (l_0_1 * omega_0_1**2)
    c5_1 = 0

    # Sensor 2
    omega_0_2 = f2 * 2 * np.pi
    h2 = 1e-6 * omega_0_2
    l_0_2 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0_2**2)
    c1_2 = beta / omega_0_2
    c2_2 = 1 / Q_0
    c3_2 = 1 / (tau * omega_0_2)
    c4_2 = (kappa * l_0_2) / u_max
    # c5_2 = mu / (l_0_2 * omega_0_2**2)
    c5_2 = 0

    h = min(h1, h2)          # more conservative (lower) time-step

    a_vals = np.linspace(-2.0, 2.0, 51)
    # g_vals = np.linspace(-1.0, 1.0, 21)
    g_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    u_dc_vals = np.linspace(-1.0, 1.0, 21)


    result_grid = run_simulation_parallel(a_vals, g_vals, u_dc_vals, N=50_000_000, h=h,
                                        c1_1=c1_1, c2_1=c2_1, c3_1=c3_1, c4_1=c4_1,
                                        c1_2=c1_2, c2_2=c2_2, c3_2=c3_2, c4_2=c4_2)
    
    np.save(f"maxima-counts-coupled-sensors-f1-{int(f1)}-f2-{int(f2)}-Q_0-{int(Q_0)}-(+++-).npy", result_grid)

    for i in range(51):
        for j in range(5):
            for k in range(21):
                result_grid[i,j,k] = min(result_grid[i,j,k], 10)

    selected_g=[-1.0, -0.5, 0.0, 0.5, 1.0]
    save_path=f"coupled_sensor_bifurcation_map_f1_{int(f1)}_f2_{int(f2)}-Q_0-{int(Q_0)}-(+++-).png"

    if selected_g is None:
        idx = np.round(np.linspace(0, len(g_vals)-1, 5)).astype(int)
        selected_g = g_vals[idx]

    n_cols = len(selected_g)
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 8), dpi=300,
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    # Discrete colours + chaos cap
    max_show = 10
    cmap_vir = mpl.colormaps['viridis']
    colors = cmap_vir(np.linspace(0, 1, max_show + 2))
    cmap = ListedColormap(colors[:max_show + 1])
    bounds = np.arange(-0.5, max_show + 1.5, 1)
    norm = BoundaryNorm(bounds, ncolors=max_show + 1)

    for col_idx, g_val in enumerate(selected_g):
        g_idx = np.argmin(np.abs(g_vals - g_val))
        data_2d = result_grid[:, g_idx, :]

        ax = axes[col_idx]
        pcm = ax.pcolormesh(u_dc_vals, a_vals, data_2d,
                            cmap=cmap, norm=norm,
                            shading='nearest', rasterized=True)

        ax.set_title(f'g = {g_val:.1f}', fontsize=30, pad=6)
        ax.set_xlabel(r'$u_{dc}$', fontsize=30)
        if col_idx == 0:
            ax.set_ylabel(r'$a$ (feedback strengh)', fontsize=30)
        ax.grid(True, alpha=0.3, ls=':')
        ax.tick_params(axis='both', which='major', labelsize=30)

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.046, pad=0.02,
                        ticks=range(max_show + 1))
    cbar.ax.tick_params(labelsize=30)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=str(i)) for i in range(6)]
    legend_elements.append(mpatches.Patch(facecolor=colors[-1], label='10+ (chaotic)'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=7,
               bbox_to_anchor=(0.5, -0.12), title='Extrema count (sensor-1)', fontsize=20)

    fig.suptitle("Effect of Coupling Strength g on Multistability and Bifurcations\n"
                 "(Coupled Sensors – 8D RK4 simulation)", fontsize=25, y=1.02)
    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()