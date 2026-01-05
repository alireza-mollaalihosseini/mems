import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def design_sensor_map(freqs, mean_mag, n_sensors=16, 
                      min_prominence_db=1.0, Q_default=18.0,
                      max_bw_hz=200.0, smooth_window_frac=0.01,
                      plot=False, log_spacing=True):
    import numpy as np
    from scipy.signal import find_peaks

    smooth_window = int(len(freqs) * smooth_window_frac)
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = np.convolve(mean_mag, np.ones(smooth_window)/smooth_window, mode='same')

    # Divide frequency range into sub-bands
    f_min, f_max = freqs[1], freqs[-1]
    if log_spacing:
        band_edges = np.logspace(np.log10(f_min+1), np.log10(f_max), n_sensors+1)
    else:
        band_edges = np.linspace(f_min, f_max, n_sensors+1)

    sensor_map = []
    for i in range(n_sensors):
        f_lo, f_hi = band_edges[i], band_edges[i+1]
        idx = (freqs >= f_lo) & (freqs < f_hi)
        if np.sum(idx) < 5:
            continue

        segment_mag = smoothed[idx]
        segment_freq = freqs[idx]

        # Find local peaks in the segment
        peaks, props = find_peaks(segment_mag, prominence=min_prominence_db)
        if len(peaks) == 0:
            # fallback: take mean freq in this band
            f0 = (f_lo + f_hi) / 2
            bw = min(max_bw_hz, (f_hi - f_lo) / 2)
            sensor_map.append({
                'f0': f0,
                'bw': bw,
                'f_low': f0 - bw/2,
                'f_high': f0 + bw/2,
                'peak_db': float(np.mean(segment_mag)),
                'method': 'fill_uniform'
            })
        else:
            # pick the most prominent peak in this band
            best_idx = np.argmax(props['prominences'])
            f0 = segment_freq[peaks[best_idx]]
            bw = min(max_bw_hz, f0 / Q_default)
            sensor_map.append({
                'f0': f0,
                'bw': bw,
                'f_low': f0 - bw/2,
                'f_high': f0 + bw/2,
                'peak_db': float(segment_mag[peaks[best_idx]]),
                'method': 'peak_detected'
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
        plt.savefig('sensor-freq-map.png', dpi=300)
        plt.close()

    return sensor_map

load_path = '/scratch/almo2783/scratch/audio-spec'

freqs = np.load(os.path.join(load_path, "Frequencies.npy"))
mean_mag = np.load(os.path.join(load_path, "Mean_magnitudes.npy"))

sensor_map = design_sensor_map(
    freqs, mean_mag,
    n_sensors=16,
    max_bw_hz=200.0,
    log_spacing=True,
    plot=True
)

for s in sensor_map:
    print(s)