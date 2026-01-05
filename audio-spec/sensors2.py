import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def design_sensor_map(f_centers, band_widths,
                      freqs, mean_mag, n_sensors=None, 
                      plot=False, log_spacing=True):

    # Use all if n_sensors not specified
    if n_sensors is None:
        n_sensors = len(f_centers)
    else:
        # Subsample evenly if fewer sensors requested (e.g., for similarity to original n=16)
        indices = np.linspace(0, len(f_centers)-1, n_sensors, dtype=int)
        f_centers = f_centers[indices]
        band_widths = band_widths[indices]

    # # Generate freqs for plotting (log-spaced, covering range with margin)
    # f_min, f_max = 0.5 * f_centers[0], 1.5 * f_centers[-1]
    # freqs = np.logspace(np.log10(f_min), np.log10(f_max), 5000)

    # Convert to dB (assuming reference 1)
    mean_mag = 20 * np.log10(np.maximum(mean_mag, 1e-10))  # Avoid log(0)

    # Smooth slightly for aesthetics (similar to original)
    smooth_window_frac = 0.01
    smooth_window = int(len(freqs) * smooth_window_frac)
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = np.convolve(mean_mag, np.ones(smooth_window)/smooth_window, mode='same')

    # Build sensor_map directly from provided values
    sensor_map = []
    for f0, bw in zip(f_centers, band_widths):
        sensor_map.append({
            'f0': f0,
            'bw': bw,
            'f_low': f0 - bw/2,
            'f_high': f0 + bw/2,
            'peak_db': float(20 * np.log10(1.0)),  # Nominal 0 dB peak
            'method': 'given'
        })

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(freqs, mean_mag, label="Mean magnitude (smoothed)")
        for s in sensor_map:
            plt.axvspan(s['f_low'], s['f_high'], color='orange', alpha=0.3)
            plt.axvline(s['f0'], color='red', lw=1)
        plt.xscale('log')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.title("Sensor frequency map")
        plt.legend()
        plt.tight_layout()
        plt.savefig('sensor-freq-map-2.png', dpi=300)
        plt.close()

    return sensor_map


# Provided info
Q_0 = 50
# f_values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1600, 2400, 3200, 4000, 5600, 7000, 9000, 12000, 14000])
f_values = np.array([50, 100, 150, 200, 300, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1600, 2400, 3000, 3500, 4000, 5600, 6000, 7000, 8000, 9000, 12000, 14000, 15000])
# band_width_values = np.maximum(f_values[1:] / Q_0, 50 * (np.diff(f_values)/(2*50)))
band_width_values = np.array([50, 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50  , 100 , 200 , 200 , 200 , 200 , 200, 200, 400 , 400,  400,  500, 500,  500,  500])
# band_width_values = np.insert(band_width_values, 0, 50)

load_path = '/scratch/almo2783/scratch/audio-spec'

freqs = np.load(os.path.join(load_path, "Frequencies.npy"))
mean_mag = np.load(os.path.join(load_path, "Mean_magnitudes.npy"))

sensor_map = design_sensor_map(
    f_values, band_width_values,
    freqs, mean_mag,
    n_sensors=None,
    log_spacing=True,
    plot=True
)

for s in sensor_map:
    print(s)