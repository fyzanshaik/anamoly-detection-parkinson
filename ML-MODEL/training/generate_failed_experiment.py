import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Settings for a "Failed" Experiment
SAMPLE_RATE = 100
DURATION = 20
NOISE_LEVEL = 2.5  # Very High Noise
FAULT_STRENGTH = 0.2  # Very Weak Fault Signal (buried in noise)

def generate_messy_signal(duration, freq, is_fault=False):
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE))
    
    # Gravity (Mean)
    signal = np.ones_like(t) * 9.8
    
    # High Random Noise
    signal += np.random.randn(len(t)) * NOISE_LEVEL
    
    # Weak Fault Pattern (Hard to detect)
    if is_fault:
        signal += np.sin(2 * np.pi * freq * t) * FAULT_STRENGTH
    else:
        # Normal motor vibration (also weak)
        signal += np.sin(2 * np.pi * 10 * t) * FAULT_STRENGTH

    return t, signal

def extract_basic_features(signal, window_size=64):
    features = []
    for i in range(0, len(signal) - window_size, window_size // 2):
        window = signal[i:i+window_size]
        
        # Poor Feature Selection (Just basic stats, no FFT, no centering)
        mean = np.mean(window)
        std = np.std(window)
        peak = np.max(window)
        low = np.min(window)
        
        features.append([mean, std, peak, low])
    return np.array(features)

def create_failed_experiment():
    output_dir = '../datasets/failed_experiment'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate Data
    t, normal_sig = generate_messy_signal(DURATION, 10, is_fault=False)
    _, bearing_sig = generate_messy_signal(DURATION, 120, is_fault=True) # 120Hz
    _, rotor_sig = generate_messy_signal(DURATION, 35, is_fault=True)    # 35Hz
    
    # 2. Extract Features
    feat_norm = extract_basic_features(normal_sig)
    feat_bear = extract_basic_features(bearing_sig)
    feat_rotor = extract_basic_features(rotor_sig)
    
    # 3. Create DataFrame
    cols = ['Mean', 'StdDev', 'Peak', 'Min']
    df1 = pd.DataFrame(feat_norm, columns=cols)
    df1['Status'] = 'Normal'
    
    df2 = pd.DataFrame(feat_bear, columns=cols)
    df2['Status'] = 'Bearing Fault'
    
    df3 = pd.DataFrame(feat_rotor, columns=cols)
    df3['Status'] = 'Rotor Imbalance'
    
    df = pd.concat([df1, df2, df3])
    df.to_csv(f'{output_dir}/failed_data.csv', index=False)

    # 4. Visualization 1: Messy Signals
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t[:500], normal_sig[:500], color='green', alpha=0.7)
    plt.title('Early Prototype: Normal Signal (High Noise)', fontweight='bold')
    plt.ylabel('Vibration')
    
    plt.subplot(3, 1, 2)
    plt.plot(t[:500], bearing_sig[:500], color='red', alpha=0.7)
    plt.title('Early Prototype: Bearing Fault (Signal Lost in Noise)', fontweight='bold')
    plt.ylabel('Vibration')
    
    plt.subplot(3, 1, 3)
    plt.plot(t[:500], rotor_sig[:500], color='blue', alpha=0.7)
    plt.title('Early Prototype: Rotor Imbalance (Indistinguishable)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Vibration')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/failed_signals.png', dpi=300)
    plt.close()

    # 5. Visualization 2: Overlapping PCA (The "Blob")
    x = df[cols].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=components[:,0], y=components[:,1], hue=df['Status'], 
                    palette={'Normal':'green', 'Bearing Fault':'red', 'Rotor Imbalance':'blue'},
                    s=80, alpha=0.6)
    
    plt.title('Initial Model Results: Complete Overlap (Failure)', fontsize=14, fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(f'{output_dir}/failed_pca.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_failed_experiment()
