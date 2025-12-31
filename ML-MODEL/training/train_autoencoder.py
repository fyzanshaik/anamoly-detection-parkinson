import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os

INPUT_DIM = 8
HIDDEN_DIM = 4
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

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

def load_data():
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    df = pd.read_csv('../datasets/combined_dataset.csv')

    X = df[['mean', 'peak', 'rms', 'skewness', 'kurtosis',
            'dominant_freq', 'harmonic_ratio', 'energy']].values
    y = df['label'].values

    print(f"\nTotal samples: {len(X)}")
    print(f"Normal samples: {np.sum(y == 0)}")
    print(f"Anomaly samples: {np.sum(y == 1)}")

    return X, y, df

def preprocess_data(X, y):
    print("\n" + "=" * 60)
    print("Preprocessing Data")
    print("=" * 60)

    X_normal = X[y == 0]
    X_anomaly = X[y == 1]

    X_train_normal, X_val_normal = train_test_split(
        X_normal, test_size=VALIDATION_SPLIT, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = scaler.transform(X_val_normal)
    X_anomaly_scaled = scaler.transform(X_anomaly)

    print(f"\nTraining samples: {len(X_train_scaled)}")
    print(f"Validation samples: {len(X_val_scaled)}")
    print(f"Anomaly samples: {len(X_anomaly_scaled)}")

    return X_train_scaled, X_val_scaled, X_anomaly_scaled, scaler

def train_model(X_train, X_val):
    print("\n" + "=" * 60)
    print("Training Autoencoder")
    print("=" * 60)
    print(f"\nArchitecture: {INPUT_DIM} → {HIDDEN_DIM} → {INPUT_DIM}")
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"Loss: Mean Squared Error")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}\n")

    model = Autoencoder(INPUT_DIM, HIDDEN_DIM)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_train, X_val, X_anomaly):
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    anomaly_pred = model.predict(X_anomaly)

    train_mse = np.mean(np.square(X_train - train_pred), axis=1)
    val_mse = np.mean(np.square(X_val - val_pred), axis=1)
    anomaly_mse = np.mean(np.square(X_anomaly - anomaly_pred), axis=1)

    threshold = np.mean(val_mse) + 2 * np.std(val_mse)

    print(f"\nReconstruction Error Statistics:")
    print(f"  Normal (train) - Mean: {np.mean(train_mse):.4f}, Std: {np.std(train_mse):.4f}")
    print(f"  Normal (val)   - Mean: {np.mean(val_mse):.4f}, Std: {np.std(val_mse):.4f}")
    print(f"  Anomaly        - Mean: {np.mean(anomaly_mse):.4f}, Std: {np.std(anomaly_mse):.4f}")
    print(f"\nRecommended Threshold: {threshold:.4f}")

    anomaly_detected = np.sum(anomaly_mse > threshold)
    accuracy = anomaly_detected / len(anomaly_mse) * 100

    false_positives = np.sum(val_mse > threshold)
    fpr = false_positives / len(val_mse) * 100

    print(f"\nPerformance Metrics:")
    print(f"  Anomaly Detection Rate: {accuracy:.2f}%")
    print(f"  False Positive Rate: {fpr:.2f}%")

    return threshold, train_mse, val_mse, anomaly_mse

def plot_results(history, train_mse, val_mse, anomaly_mse, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(train_mse, bins=50, alpha=0.7, label='Normal (train)', color='green')
    axes[1].hist(val_mse, bins=50, alpha=0.7, label='Normal (val)', color='blue')
    axes[1].hist(anomaly_mse, bins=50, alpha=0.7, label='Anomaly', color='red')
    axes[1].axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    axes[1].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Reconstruction Error (MSE)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../models/training_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: ../models/training_results.png")
    plt.close()

def export_model(model, scaler, threshold):
    print("\n" + "=" * 60)
    print("Exporting Model")
    print("=" * 60)

    os.makedirs('../models', exist_ok=True)

    encoder_weights = model.encoder.layers[0].get_weights()[0]
    encoder_bias = model.encoder.layers[0].get_weights()[1]
    decoder_weights = model.decoder.get_weights()[0]
    decoder_bias = model.decoder.get_weights()[1]

    np.save('../models/encoder_weights.npy', encoder_weights)
    np.save('../models/encoder_bias.npy', encoder_bias)
    np.save('../models/decoder_weights.npy', decoder_weights)
    np.save('../models/decoder_bias.npy', decoder_bias)
    np.save('../models/scaler_mean.npy', scaler.mean_)
    np.save('../models/scaler_std.npy', scaler.scale_)

    config = {
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'threshold': float(threshold),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist()
    }

    with open('../models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nModel weights saved to: ../models/")
    print(f"  - encoder_weights.npy")
    print(f"  - encoder_bias.npy")
    print(f"  - decoder_weights.npy")
    print(f"  - decoder_bias.npy")
    print(f"  - scaler_mean.npy")
    print(f"  - scaler_std.npy")
    print(f"  - model_config.json")

def main():
    X, y, df = load_data()
    X_train, X_val, X_anomaly, scaler = preprocess_data(X, y)
    model, history = train_model(X_train, X_val)
    threshold, train_mse, val_mse, anomaly_mse = evaluate_model(model, X_train, X_val, X_anomaly)
    plot_results(history, train_mse, val_mse, anomaly_mse, threshold)
    export_model(model, scaler, threshold)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\n✅ Model ready for deployment to ESP32\n")

if __name__ == "__main__":
    main()
