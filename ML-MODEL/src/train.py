import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras

from model import build_autoencoder, MODEL_CONFIGS
from features import FEATURE_GROUPS, ALL_FEATURE_NAMES, STAT_FEATURE_NAMES, FREQ_FEATURE_NAMES

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def get_feature_slice(feature_group):
    if feature_group == "all":
        return slice(None), ALL_FEATURE_NAMES
    elif feature_group == "stat_freq":
        n_stat = len(STAT_FEATURE_NAMES)
        n_freq = len(FREQ_FEATURE_NAMES)
        names = STAT_FEATURE_NAMES + FREQ_FEATURE_NAMES
        indices = list(range(n_stat)) + list(range(n_stat, n_stat + n_freq))
        return indices, names
    else:
        names, sl = FEATURE_GROUPS[feature_group]
        return sl, names


def train_single_config(config_name, X, y, fault_types):
    config = MODEL_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")

    feat_slice, feat_names = get_feature_slice(config["feature_group"])
    X_subset = X[:, feat_slice]
    input_dim = X_subset.shape[1]

    print(f"  Feature group: {config['feature_group']}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, config['hidden_dims']))} -> {input_dim}")

    X_normal = X_subset[y == 0]
    X_anomaly = X_subset[y == 1]

    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_anomaly_scaled = scaler.transform(X_anomaly)

    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Validation samples: {len(X_val_scaled)}")
    print(f"  Anomaly test samples: {len(X_anomaly_scaled)}")

    autoencoder, encoder, decoder = build_autoencoder(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
    )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
    )

    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(X_val_scaled, X_val_scaled),
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = min(history.history["val_loss"])
    print(f"  Best epoch: {best_epoch}, Val loss: {best_val_loss:.6f}")

    train_pred = autoencoder.predict(X_train_scaled, verbose=0)
    val_pred = autoencoder.predict(X_val_scaled, verbose=0)
    anomaly_pred = autoencoder.predict(X_anomaly_scaled, verbose=0)

    train_mse = np.mean((X_train_scaled - train_pred) ** 2, axis=1)
    val_mse = np.mean((X_val_scaled - val_pred) ** 2, axis=1)
    anomaly_mse = np.mean((X_anomaly_scaled - anomaly_pred) ** 2, axis=1)

    threshold = np.percentile(val_mse, 95)

    anomaly_detected = np.sum(anomaly_mse > threshold)
    detection_rate = anomaly_detected / len(anomaly_mse) * 100
    false_positives = np.sum(val_mse > threshold)
    fpr = false_positives / len(val_mse) * 100

    print(f"  Threshold (95th pct): {threshold:.6f}")
    print(f"  Detection rate: {detection_rate:.2f}%")
    print(f"  False positive rate: {fpr:.2f}%")

    model_dir = os.path.join(OUTPUT_DIR, "models", config_name)
    os.makedirs(model_dir, exist_ok=True)

    autoencoder.save(os.path.join(model_dir, "autoencoder.keras"))

    np.save(os.path.join(model_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(model_dir, "scaler_scale.npy"), scaler.scale_)

    results = {
        "config_name": config_name,
        "feature_group": config["feature_group"],
        "feature_names": feat_names if isinstance(feat_names, list) else list(feat_names),
        "input_dim": input_dim,
        "hidden_dims": config["hidden_dims"],
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "threshold": float(threshold),
        "detection_rate": float(detection_rate),
        "false_positive_rate": float(fpr),
        "n_train": len(X_train_scaled),
        "n_val": len(X_val_scaled),
        "n_anomaly": len(X_anomaly_scaled),
        "train_mse_mean": float(np.mean(train_mse)),
        "train_mse_std": float(np.std(train_mse)),
        "val_mse_mean": float(np.mean(val_mse)),
        "val_mse_std": float(np.std(val_mse)),
        "anomaly_mse_mean": float(np.mean(anomaly_mse)),
        "anomaly_mse_std": float(np.std(anomaly_mse)),
    }

    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return {
        "results": results,
        "history": history.history,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "anomaly_mse": anomaly_mse,
        "threshold": threshold,
        "scaler": scaler,
        "autoencoder": autoencoder,
        "encoder": encoder,
        "decoder": decoder,
        "X_val_scaled": X_val_scaled,
        "X_anomaly_scaled": X_anomaly_scaled,
        "feat_slice": feat_slice,
    }


def train_all_configs(X, y, fault_types, configs=None):
    if configs is None:
        configs = list(MODEL_CONFIGS.keys())

    all_results = {}
    for config_name in configs:
        all_results[config_name] = train_single_config(config_name, X, y, fault_types)

    summary_path = os.path.join(OUTPUT_DIR, "metrics", "training_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    summary = {}
    for name, res in all_results.items():
        summary[name] = res["results"]

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Input':>5} {'Detect%':>8} {'FPR%':>6} {'Threshold':>10}")
    print("-" * 50)
    for name, res in all_results.items():
        r = res["results"]
        print(f"{name:<15} {r['input_dim']:>5} {r['detection_rate']:>7.1f}% {r['false_positive_rate']:>5.1f}% {r['threshold']:>10.4f}")

    return all_results
