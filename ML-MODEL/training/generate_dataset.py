import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

SAMPLE_RATE = 100
DURATION_NORMAL = 100
DURATION_FAULT = 20
NOISE_LEVEL = 0.5

def generate_normal_vibration(duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate))
    base_freq = 60.0

    signal = (
        2.0 * np.sin(2 * np.pi * base_freq * t) +
        0.5 * np.sin(2 * np.pi * 2 * base_freq * t) +
        0.3 * np.sin(2 * np.pi * 3 * base_freq * t) +
        NOISE_LEVEL * np.random.randn(len(t))
    )

    modulation = 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    signal = signal * modulation
    signal = signal + 9.8

    return t, signal

def generate_bearing_fault(duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate))
    base_freq = 60.0
    fault_freq = 120.0

    signal = (
        2.0 * np.sin(2 * np.pi * base_freq * t) +
        0.5 * np.sin(2 * np.pi * 2 * base_freq * t) +
        3.0 * np.sin(2 * np.pi * fault_freq * t) +
        1.5 * np.sin(2 * np.pi * 2 * fault_freq * t) +
        NOISE_LEVEL * np.random.randn(len(t))
    )

    modulation = 1.0 + 0.15 * np.sin(2 * np.pi * 0.5 * t)
    signal = signal * modulation + 9.8

    return t, signal

def generate_rotor_imbalance(duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate))
    base_freq = 60.0
    imbalance_freq = 35.0

    signal = (
        2.0 * np.sin(2 * np.pi * base_freq * t) +
        0.5 * np.sin(2 * np.pi * 2 * base_freq * t) +
        2.5 * np.sin(2 * np.pi * imbalance_freq * t) +
        1.0 * np.sin(2 * np.pi * 2 * imbalance_freq * t) +
        NOISE_LEVEL * np.random.randn(len(t))
    )

    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
    signal = signal * modulation + 9.8

    return t, signal

def extract_features(signal, window_size=64):
    features = []

    for i in range(0, len(signal) - window_size, window_size // 2):
        window = signal[i:i+window_size]

        mean = np.mean(window)
        peak = np.max(window)
        rms = np.sqrt(np.mean(window**2))

        skewness = np.mean(((window - mean) / np.std(window))**3) if np.std(window) > 0 else 0
        kurtosis = np.mean(((window - mean) / np.std(window))**4) if np.std(window) > 0 else 0

        # Remove DC component (gravity) for FFT analysis
        window_centered = window - mean
        fft = np.fft.fft(window_centered)
        freqs = np.fft.fftfreq(len(window), 1/SAMPLE_RATE)
        
        # Find peak frequency (ignoring 0Hz index since we centered it, but argmax might still pick small noise near 0)
        # We look at the positive half of the spectrum
        positive_freqs = freqs[:len(window)//2]
        positive_fft = np.abs(fft[:len(window)//2])
        
        # Skip the first bin (0Hz) just in case
        dominant_freq = abs(positive_freqs[np.argmax(positive_fft[1:]) + 1])

        harmonic_ratio = peak / mean if mean > 0 else 0
        energy = np.sum(window**2) / len(window)

        features.append([mean, peak, rms, skewness, kurtosis, dominant_freq, harmonic_ratio, energy])

    return np.array(features)

def create_dataset():
    print("=" * 60)
    print("Generating Vibration Dataset for Anomaly Detection")
    print("=" * 60)

    print("\n[1/4] Generating normal vibration data...")
    t_normal, signal_normal = generate_normal_vibration(DURATION_NORMAL, SAMPLE_RATE)

    print("[2/4] Generating bearing fault data...")
    t_bearing, signal_bearing = generate_bearing_fault(DURATION_FAULT, SAMPLE_RATE)

    print("[3/4] Generating rotor imbalance data...")
    t_rotor, signal_rotor = generate_rotor_imbalance(DURATION_FAULT, SAMPLE_RATE)

    print("[4/4] Extracting features...")
    features_normal = extract_features(signal_normal)
    features_bearing = extract_features(signal_bearing)
    features_rotor = extract_features(signal_rotor)

    df_normal = pd.DataFrame(features_normal,
                             columns=['mean', 'peak', 'rms', 'skewness', 'kurtosis',
                                     'dominant_freq', 'harmonic_ratio', 'energy'])
    df_normal['label'] = 0
    df_normal['fault_type'] = 'normal'

    df_bearing = pd.DataFrame(features_bearing,
                              columns=['mean', 'peak', 'rms', 'skewness', 'kurtosis',
                                      'dominant_freq', 'harmonic_ratio', 'energy'])
    df_bearing['label'] = 1
    df_bearing['fault_type'] = 'bearing_fault'

    df_rotor = pd.DataFrame(features_rotor,
                            columns=['mean', 'peak', 'rms', 'skewness', 'kurtosis',
                                    'dominant_freq', 'harmonic_ratio', 'energy'])
    df_rotor['label'] = 1
    df_rotor['fault_type'] = 'rotor_imbalance'

    df_combined = pd.concat([df_normal, df_bearing, df_rotor], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    datasets_dir = '../datasets'
    os.makedirs(datasets_dir, exist_ok=True)

    df_normal.to_csv(f'{datasets_dir}/normal_vibration.csv', index=False)
    df_bearing.to_csv(f'{datasets_dir}/bearing_fault.csv', index=False)
    df_rotor.to_csv(f'{datasets_dir}/rotor_imbalance.csv', index=False)
    df_combined.to_csv(f'{datasets_dir}/combined_dataset.csv', index=False)

    np.save(f'{datasets_dir}/signal_normal.npy', signal_normal)
    np.save(f'{datasets_dir}/signal_bearing.npy', signal_bearing)
    np.save(f'{datasets_dir}/signal_rotor.npy', signal_rotor)

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Normal samples:         {len(df_normal)}")
    print(f"Bearing fault samples:  {len(df_bearing)}")
    print(f"Rotor imbalance samples: {len(df_rotor)}")
    print(f"Total samples:          {len(df_combined)}")
    print(f"\nClass distribution:")
    print(df_combined['label'].value_counts())
    print(f"\nFault type distribution:")
    print(df_combined['fault_type'].value_counts())
    print(f"\nFeature statistics (first 5 rows):")
    print(df_combined.head())
    print(f"\nDatasets saved to: {os.path.abspath(datasets_dir)}")

    plot_signals(signal_normal[:1000], signal_bearing[:1000], signal_rotor[:1000])

    return df_combined

def plot_signals(normal, bearing, rotor):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    t = np.linspace(0, 10, 1000)

    axes[0].plot(t, normal, 'g-', linewidth=0.5)
    axes[0].set_title('Normal Vibration', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Vibration (m/s²)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, bearing, 'r-', linewidth=0.5)
    axes[1].set_title('Bearing Fault (120Hz spike)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Vibration (m/s²)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, rotor, 'b-', linewidth=0.5)
    axes[2].set_title('Rotor Imbalance (35Hz spike)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Vibration (m/s²)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../datasets/signal_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: ../datasets/signal_comparison.png")
    plt.close()

if __name__ == "__main__":
    df = create_dataset()
    print("\n✅ Dataset generation complete!\n")
