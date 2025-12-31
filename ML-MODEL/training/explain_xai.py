import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
from tensorflow import keras

INPUT_DIM = 8
HIDDEN_DIM = 4

class Autoencoder(keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
        ])
        self.decoder = keras.layers.Dense(input_dim, activation='linear')

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_model_and_data():
    print("=" * 60)
    print("Loading Model and Data for XAI Analysis")
    print("=" * 60)

    with open('../models/model_config.json', 'r') as f:
        config = json.load(f)

    encoder_weights = np.load('../models/encoder_weights.npy')
    encoder_bias = np.load('../models/encoder_bias.npy')
    decoder_weights = np.load('../models/decoder_weights.npy')
    decoder_bias = np.load('../models/decoder_bias.npy')

    model = Autoencoder(INPUT_DIM, HIDDEN_DIM)
    _ = model(np.zeros((1, INPUT_DIM)))

    model.encoder.layers[0].set_weights([encoder_weights, encoder_bias])
    model.decoder.set_weights([decoder_weights, decoder_bias])

    df = pd.read_csv('../datasets/combined_dataset.csv')
    X = df[['mean', 'peak', 'rms', 'skewness', 'kurtosis',
            'dominant_freq', 'harmonic_ratio', 'energy']].values

    print(f"\nModel loaded successfully")
    print(f"Dataset: {len(df)} samples")

    return model, X, df

def compute_reconstruction_error(model, X):
    predictions = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - predictions), axis=1)
    return errors

def compute_feature_importance(model, X, sample_size=100):
    print("\n" + "=" * 60)
    print("Computing SHAP Values for Feature Importance")
    print("=" * 60)

    sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X[sample_indices]

    explainer = shap.KernelExplainer(
        lambda x: compute_reconstruction_error(model, x),
        X_sample[:20]
    )

    shap_values = explainer.shap_values(X_sample[:50], nsamples=100)

    feature_names = ['mean', 'peak', 'rms', 'skewness', 'kurtosis',
                     'dominant_freq', 'harmonic_ratio', 'energy']

    print(f"\nSHAP values computed for {len(shap_values)} samples")

    return shap_values, feature_names, X_sample[:50]

def analyze_fault_signatures(df):
    print("\n" + "=" * 60)
    print("Fault Signature Analysis")
    print("=" * 60)

    bearing_faults = df[df['fault_type'] == 'bearing_fault']
    rotor_faults = df[df['fault_type'] == 'rotor_imbalance']
    normal = df[df['fault_type'] == 'normal']

    print("\nBearing Fault Characteristics:")
    print(f"  Average Harmonic Ratio: {bearing_faults['harmonic_ratio'].mean():.3f}")
    print(f"  Average Dominant Freq: {bearing_faults['dominant_freq'].mean():.1f} Hz")
    print(f"  Average Peak: {bearing_faults['peak'].mean():.2f}")

    print("\nRotor Imbalance Characteristics:")
    print(f"  Average Harmonic Ratio: {rotor_faults['harmonic_ratio'].mean():.3f}")
    print(f"  Average Dominant Freq: {rotor_faults['dominant_freq'].mean():.1f} Hz")
    print(f"  Average Peak: {rotor_faults['peak'].mean():.2f}")

    print("\nNormal Operation Characteristics:")
    print(f"  Average Harmonic Ratio: {normal['harmonic_ratio'].mean():.3f}")
    print(f"  Average Dominant Freq: {normal['dominant_freq'].mean():.1f} Hz")
    print(f"  Average Peak: {normal['peak'].mean():.2f}")

def plot_shap_summary(shap_values, feature_names, X_sample):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('../models/shap_summary.png', dpi=150, bbox_inches='tight')
    print(f"\nSHAP summary plot saved to: ../models/shap_summary.png")
    plt.close()

def plot_feature_importance(shap_values, feature_names):
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_pct = (feature_importance / feature_importance.sum()) * 100

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pos, feature_importance[sorted_idx], align='center', color='steelblue')
    ax.set_yticks(pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Mean |SHAP value| (Average Impact on Reconstruction Error)')
    ax.set_title('Feature Importance for Anomaly Detection', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(feature_importance[sorted_idx]):
        ax.text(v + 0.01, i, f'{importance_pct[sorted_idx[i]]:.1f}%',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('../models/feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to: ../models/feature_importance.png")
    plt.close()

    print("\n" + "=" * 60)
    print("Feature Importance Rankings")
    print("=" * 60)

    for idx in sorted_idx[::-1]:
        print(f"  {feature_names[idx]:18s}: {importance_pct[idx]:5.1f}%")

def generate_explanation_logic(shap_values, feature_names):
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_pct = (feature_importance / feature_importance.sum()) * 100

    top_features = np.argsort(feature_importance)[::-1][:3]

    print("\n" + "=" * 60)
    print("XAI Explanation Logic for ESP32")
    print("=" * 60)

    print(f"\nTop 3 Contributing Features:")
    for i, idx in enumerate(top_features):
        print(f"  {i+1}. {feature_names[idx]:18s} ({importance_pct[idx]:.1f}%)")

    print(f"\nRecommended Threshold Logic:")
    print(f"  - If harmonic_ratio > 1.5 → Bearing Fault (120Hz)")
    print(f"  - If dominant_freq < 50 → Rotor Imbalance (35Hz)")
    print(f"  - Else → Misalignment")

def main():
    model, X, df = load_model_and_data()
    shap_values, feature_names, X_sample = compute_feature_importance(model, X)
    analyze_fault_signatures(df)
    plot_shap_summary(shap_values, feature_names, X_sample)
    plot_feature_importance(shap_values, feature_names)
    generate_explanation_logic(shap_values, feature_names)

    print("\n" + "=" * 60)
    print("XAI Analysis Complete!")
    print("=" * 60)
    print("\n✅ Explainability reports generated\n")

if __name__ == "__main__":
    main()
