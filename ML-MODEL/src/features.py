import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch
import librosa

SAMPLING_RATE = 12000  # CWRU drive-end sampling rate


def statistical_features(window):
    mean = np.mean(window)
    peak = np.max(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    std = np.std(window)
    skewness = sp_stats.skew(window)
    kurtosis = sp_stats.kurtosis(window)
    crest_factor = peak / rms if rms > 1e-10 else 0.0
    shape_factor = rms / (np.mean(np.abs(window)) + 1e-10)
    impulse_factor = peak / (np.mean(np.abs(window)) + 1e-10)
    clearance_factor = peak / (np.mean(np.sqrt(np.abs(window))) ** 2 + 1e-10)

    return np.array([
        mean, peak, rms, std, skewness, kurtosis,
        crest_factor, shape_factor, impulse_factor, clearance_factor
    ], dtype=np.float64)


STAT_FEATURE_NAMES = [
    "mean", "peak", "rms", "std", "skewness", "kurtosis",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor"
]


def frequency_features(window, sr=SAMPLING_RATE):
    freqs, psd = welch(window, fs=sr, nperseg=min(256, len(window)))

    total_energy = np.sum(psd)
    dominant_freq = freqs[np.argmax(psd)]
    mean_freq = np.sum(freqs * psd) / (total_energy + 1e-10)

    psd_norm = psd / (total_energy + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    cumsum = np.cumsum(psd)
    median_idx = np.searchsorted(cumsum, total_energy / 2)
    median_freq = freqs[min(median_idx, len(freqs) - 1)]

    band_edges = [0, 500, 1500, 3000, 6000]
    band_powers = []
    for i in range(len(band_edges) - 1):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        band_powers.append(np.sum(psd[mask]) / (total_energy + 1e-10))

    return np.array([
        dominant_freq, mean_freq, median_freq, spectral_entropy, total_energy,
        *band_powers
    ], dtype=np.float64)


FREQ_FEATURE_NAMES = [
    "dominant_freq", "mean_freq", "median_freq", "spectral_entropy", "total_energy",
    "band_0_500", "band_500_1500", "band_1500_3000", "band_3000_6000"
]


def mfcc_features(window, sr=SAMPLING_RATE, n_mfcc=13):
    window_float = window.astype(np.float32)
    mfccs = librosa.feature.mfcc(y=window_float, sr=sr, n_mfcc=n_mfcc, n_fft=min(512, len(window)))

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    n_frames = mfccs.shape[1]
    if n_frames >= 9:
        mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
    elif n_frames >= 3:
        mfcc_delta = np.mean(librosa.feature.delta(mfccs, width=min(n_frames, 3)), axis=1)
    else:
        mfcc_delta = np.zeros(n_mfcc)

    return np.concatenate([mfcc_mean, mfcc_std, mfcc_delta]).astype(np.float64)


MFCC_FEATURE_NAMES = (
    [f"mfcc_{i}_mean" for i in range(13)] +
    [f"mfcc_{i}_std" for i in range(13)] +
    [f"mfcc_{i}_delta" for i in range(13)]
)


def extract_all_features(window, sr=SAMPLING_RATE):
    stat = statistical_features(window)
    freq = frequency_features(window, sr)
    mfcc = mfcc_features(window, sr)
    return np.concatenate([stat, freq, mfcc])


ALL_FEATURE_NAMES = STAT_FEATURE_NAMES + FREQ_FEATURE_NAMES + MFCC_FEATURE_NAMES

FEATURE_GROUPS = {
    "statistical": (STAT_FEATURE_NAMES, slice(0, len(STAT_FEATURE_NAMES))),
    "frequency": (FREQ_FEATURE_NAMES, slice(
        len(STAT_FEATURE_NAMES),
        len(STAT_FEATURE_NAMES) + len(FREQ_FEATURE_NAMES)
    )),
    "mfcc": (MFCC_FEATURE_NAMES, slice(
        len(STAT_FEATURE_NAMES) + len(FREQ_FEATURE_NAMES),
        len(ALL_FEATURE_NAMES)
    )),
}


def extract_windows(signal, window_size=2048, hop_size=1024):
    windows = []
    for start in range(0, len(signal) - window_size + 1, hop_size):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


def extract_features_from_signal(signal, sr=SAMPLING_RATE, window_size=2048, hop_size=1024):
    windows = extract_windows(signal, window_size, hop_size)
    features = np.array([extract_all_features(w, sr) for w in windows])
    return features


def extract_dataset_features(signals_dict, sr=SAMPLING_RATE, window_size=2048, hop_size=1024):
    all_features = []
    all_labels = []
    all_fault_types = []
    all_loads = []
    all_keys = []

    for key, info in signals_dict.items():
        signal = info["signal"]
        fault_type = info["fault_type"]
        load = info["load"]

        label = 0 if fault_type == "normal" else 1

        features = extract_features_from_signal(signal, sr, window_size, hop_size)
        n_windows = features.shape[0]

        all_features.append(features)
        all_labels.extend([label] * n_windows)
        all_fault_types.extend([fault_type] * n_windows)
        all_loads.extend([load] * n_windows)
        all_keys.extend([key] * n_windows)

        print(f"  {key}: {n_windows} windows, {features.shape[1]} features")

    X = np.vstack(all_features)
    y = np.array(all_labels)
    fault_types = np.array(all_fault_types)
    loads = np.array(all_loads)
    keys = np.array(all_keys)

    return X, y, fault_types, loads, keys
