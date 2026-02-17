import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import librosa
import librosa.display

from features import SAMPLING_RATE, STAT_FEATURE_NAMES, FREQ_FEATURE_NAMES, MFCC_FEATURE_NAMES

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "metrics")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def plot_raw_signals(signals_dict, n_samples=6000):
    ensure_dirs()
    fault_examples = {}
    for key, info in signals_dict.items():
        ft = info["fault_type"]
        if ft not in fault_examples:
            fault_examples[ft] = info["signal"][:n_samples]

    n_types = len(fault_examples)
    fig, axes = plt.subplots(n_types, 1, figsize=(14, 3 * n_types), sharex=True)
    if n_types == 1:
        axes = [axes]

    colors = {"normal": "#2ecc71", "inner_race_007": "#e74c3c", "inner_race_014": "#c0392b",
              "inner_race_021": "#a93226", "ball_007": "#3498db", "ball_014": "#2980b9",
              "ball_021": "#1f618d", "outer_race_007": "#f39c12", "outer_race_014": "#e67e22",
              "outer_race_021": "#d35400"}

    t = np.arange(n_samples) / SAMPLING_RATE

    for i, (ft, signal) in enumerate(fault_examples.items()):
        color = colors.get(ft, "#95a5a6")
        axes[i].plot(t, signal, color=color, linewidth=0.3, alpha=0.8)
        axes[i].set_ylabel("Acceleration (g)")
        label = ft.replace("_", " ").title()
        axes[i].set_title(f"{label}", fontweight="bold")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw Vibration Signals by Fault Type", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_raw_signals.png"))
    plt.close()
    print("  Saved: 01_raw_signals.png")


def plot_mfcc_spectrograms(signals_dict, n_samples=12000):
    ensure_dirs()
    fault_examples = {}
    for key, info in signals_dict.items():
        ft = info["fault_type"]
        if ft not in fault_examples:
            fault_examples[ft] = info["signal"][:n_samples].astype(np.float32)

    n_types = len(fault_examples)
    fig, axes = plt.subplots(n_types, 1, figsize=(14, 3 * n_types))
    if n_types == 1:
        axes = [axes]

    for i, (ft, signal) in enumerate(fault_examples.items()):
        mfccs = librosa.feature.mfcc(y=signal, sr=SAMPLING_RATE, n_mfcc=13, n_fft=512, hop_length=256)
        img = librosa.display.specshow(mfccs, x_axis="time", sr=SAMPLING_RATE,
                                       hop_length=256, ax=axes[i], cmap="coolwarm")
        label = ft.replace("_", " ").title()
        axes[i].set_title(f"MFCC - {label}", fontweight="bold")
        axes[i].set_ylabel("MFCC Coefficient")
        fig.colorbar(img, ax=axes[i], format="%+2.0f")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("MFCC Spectrograms by Fault Type", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_mfcc_spectrograms.png"))
    plt.close()
    print("  Saved: 02_mfcc_spectrograms.png")


def plot_feature_distributions(X, y, fault_types):
    ensure_dirs()
    stat_features = X[:, :len(STAT_FEATURE_NAMES)]

    n_feats = min(6, stat_features.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_feats):
        ax = axes[i]
        for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Anomaly")]:
            data = stat_features[y == label, i]
            ax.hist(data, bins=50, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(STAT_FEATURE_NAMES[i].replace("_", " ").title(), fontweight="bold")
        ax.legend()
        ax.set_ylabel("Density")

    fig.suptitle("Statistical Feature Distributions: Normal vs Anomaly", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_feature_distributions.png"))
    plt.close()
    print("  Saved: 03_feature_distributions.png")


def plot_tsne_clusters(X, y, fault_types):
    ensure_dirs()
    n_samples = min(3000, len(X))
    indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
    X_sub = X[indices]
    ft_sub = fault_types[indices]

    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X_sub)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)

    unique_faults = np.unique(ft_sub)
    colors_map = {
        "normal": "#2ecc71", "inner_race_007": "#e74c3c", "inner_race_014": "#c0392b",
        "inner_race_021": "#a93226", "ball_007": "#3498db", "ball_014": "#2980b9",
        "ball_021": "#1f618d", "outer_race_007": "#f39c12", "outer_race_014": "#e67e22",
        "outer_race_021": "#d35400"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ft in unique_faults:
        mask = ft_sub == ft
        color = colors_map.get(ft, "#95a5a6")
        label = ft.replace("_", " ").title()
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=label,
                        alpha=0.6, s=10, edgecolors="none")

    axes[0].set_title("t-SNE by Fault Type", fontweight="bold")
    axes[0].legend(markerscale=3, fontsize=8)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    y_sub = np.array([0 if ft == "normal" else 1 for ft in ft_sub])
    for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Anomaly")]:
        mask = y_sub == label
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                        alpha=0.6, s=10, edgecolors="none")

    axes[1].set_title("t-SNE by Class (Normal vs Anomaly)", fontweight="bold")
    axes[1].legend(markerscale=3)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    fig.suptitle("Feature Space Visualization (t-SNE)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_tsne_clusters.png"))
    plt.close()
    print("  Saved: 04_tsne_clusters.png")


def plot_training_curves(all_results):
    ensure_dirs()
    n_configs = len(all_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 5))
    if n_configs == 1:
        axes = [axes]

    for i, (name, res) in enumerate(all_results.items()):
        history = res["history"]
        axes[i].plot(history["loss"], label="Train Loss", color="#3498db")
        axes[i].plot(history["val_loss"], label="Val Loss", color="#e74c3c")
        axes[i].set_title(f"{name}", fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("MSE Loss")
        axes[i].legend()
        axes[i].set_yscale("log")

    fig.suptitle("Training Loss Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_training_curves.png"))
    plt.close()
    print("  Saved: 05_training_curves.png")


def plot_reconstruction_error_distributions(all_results):
    ensure_dirs()
    n_configs = len(all_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 5))
    if n_configs == 1:
        axes = [axes]

    for i, (name, res) in enumerate(all_results.items()):
        ax = axes[i]
        ax.hist(res["val_mse"], bins=60, alpha=0.7, color="#2ecc71", label="Normal", density=True)
        ax.hist(res["anomaly_mse"], bins=60, alpha=0.7, color="#e74c3c", label="Anomaly", density=True)
        ax.axvline(res["threshold"], color="black", linestyle="--", linewidth=2,
                   label=f"Threshold: {res['threshold']:.4f}")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Reconstruction Error Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_reconstruction_errors.png"))
    plt.close()
    print("  Saved: 06_reconstruction_errors.png")


def plot_roc_curves(all_results):
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    metrics = {}

    for i, (name, res) in enumerate(all_results.items()):
        scores = np.concatenate([res["val_mse"], res["anomaly_mse"]])
        labels = np.concatenate([np.zeros(len(res["val_mse"])), np.ones(len(res["anomaly_mse"]))])

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f"{name} (AUC={roc_auc:.4f})")

        metrics[name] = {"auc_roc": float(roc_auc)}

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_roc_curves.png"))
    plt.close()
    print("  Saved: 07_roc_curves.png")

    with open(os.path.join(METRICS_DIR, "roc_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_precision_recall_curves(all_results):
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, (name, res) in enumerate(all_results.items()):
        scores = np.concatenate([res["val_mse"], res["anomaly_mse"]])
        labels = np.concatenate([np.zeros(len(res["val_mse"])), np.ones(len(res["anomaly_mse"]))])

        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)

        ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_precision_recall.png"))
    plt.close()
    print("  Saved: 08_precision_recall.png")


def plot_confusion_matrices(all_results):
    ensure_dirs()
    n_configs = len(all_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 5))
    if n_configs == 1:
        axes = [axes]

    for i, (name, res) in enumerate(all_results.items()):
        val_pred = (res["val_mse"] > res["threshold"]).astype(int)
        anomaly_pred = (res["anomaly_mse"] > res["threshold"]).astype(int)

        y_true = np.concatenate([np.zeros(len(val_pred)), np.ones(len(anomaly_pred))])
        y_pred = np.concatenate([val_pred, anomaly_pred])

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        axes[i].set_title(f"{name}", fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_confusion_matrices.png"))
    plt.close()
    print("  Saved: 09_confusion_matrices.png")


def plot_detection_latency(all_results, X, y, fault_types):
    ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    window_duration_ms = 2048 / SAMPLING_RATE * 1000
    hop_duration_ms = 1024 / SAMPLING_RATE * 1000

    for i, (name, res) in enumerate(all_results.items()):
        thresholds = np.linspace(
            min(np.min(res["val_mse"]), np.min(res["anomaly_mse"])),
            np.percentile(res["anomaly_mse"], 99),
            50
        )
        detection_rates = []
        latencies_mean = []

        for thr in thresholds:
            detected = res["anomaly_mse"] > thr
            rate = np.mean(detected) * 100
            detection_rates.append(rate)

            if np.any(detected):
                first_detection_indices = []
                chunk_size = 50
                for start in range(0, len(detected), chunk_size):
                    chunk = detected[start:start + chunk_size]
                    if np.any(chunk):
                        idx = np.argmax(chunk)
                        first_detection_indices.append(idx)
                if first_detection_indices:
                    mean_latency = np.mean(first_detection_indices) * hop_duration_ms
                else:
                    mean_latency = float("inf")
            else:
                mean_latency = float("inf")

            latencies_mean.append(mean_latency)

        axes[0].plot(thresholds, detection_rates, color=colors[i % len(colors)],
                     linewidth=2, label=name)

        finite_mask = np.isfinite(latencies_mean)
        if np.any(finite_mask):
            axes[1].plot(np.array(thresholds)[finite_mask],
                         np.array(latencies_mean)[finite_mask],
                         color=colors[i % len(colors)], linewidth=2, label=name)

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Detection Rate (%)")
    axes[0].set_title("Detection Rate vs Threshold", fontweight="bold")
    axes[0].legend()

    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Mean Latency (ms)")
    axes[1].set_title("Detection Latency vs Threshold", fontweight="bold")
    axes[1].legend()

    fig.suptitle("Detection Latency Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_detection_latency.png"))
    plt.close()
    print("  Saved: 10_detection_latency.png")


def plot_model_comparison_table(all_results):
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(12, 2 + len(all_results) * 0.6))
    ax.axis("off")

    headers = ["Model", "Input Dim", "Architecture", "AUC-ROC", "Detection %", "FPR %", "Threshold"]
    rows = []

    for name, res in all_results.items():
        r = res["results"]
        scores = np.concatenate([res["val_mse"], res["anomaly_mse"]])
        labels = np.concatenate([np.zeros(len(res["val_mse"])), np.ones(len(res["anomaly_mse"]))])
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr_arr, tpr_arr)

        arch = f"{r['input_dim']}->{'-'.join(map(str, r['hidden_dims']))}->{r['input_dim']}"
        rows.append([
            name, str(r["input_dim"]), arch,
            f"{roc_auc:.4f}", f"{r['detection_rate']:.1f}",
            f"{r['false_positive_rate']:.1f}", f"{r['threshold']:.4f}"
        ])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#ecf0f1")

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "11_model_comparison.png"))
    plt.close()
    print("  Saved: 11_model_comparison.png")


def plot_per_fault_analysis(all_results, X, y, fault_types):
    ensure_dirs()
    best_name = max(all_results.keys(), key=lambda k: all_results[k]["results"]["detection_rate"])
    best = all_results[best_name]

    feat_slice = best["feat_slice"]
    scaler_mean = best["scaler"].mean_
    scaler_scale = best["scaler"].scale_
    autoencoder = best["autoencoder"]
    threshold = best["threshold"]

    unique_faults = [ft for ft in np.unique(fault_types) if ft != "normal"]
    n_faults = len(unique_faults)

    fig, axes = plt.subplots(1, n_faults, figsize=(5 * n_faults, 5))
    if n_faults == 1:
        axes = [axes]

    for i, ft in enumerate(unique_faults):
        mask = fault_types == ft
        X_ft = X[mask][:, feat_slice]
        X_ft_scaled = (X_ft - scaler_mean) / scaler_scale
        pred = autoencoder.predict(X_ft_scaled, verbose=0)
        mse = np.mean((X_ft_scaled - pred) ** 2, axis=1)

        axes[i].hist(mse, bins=40, alpha=0.7, color="#e74c3c", density=True)
        axes[i].axvline(threshold, color="black", linestyle="--", linewidth=2,
                        label=f"Threshold: {threshold:.4f}")
        detected = np.sum(mse > threshold) / len(mse) * 100
        label = ft.replace("_", " ").title()
        axes[i].set_title(f"{label}\n(Detection: {detected:.1f}%)", fontweight="bold")
        axes[i].set_xlabel("Reconstruction Error")
        axes[i].set_ylabel("Density")
        axes[i].legend()

    fig.suptitle(f"Per-Fault Detection Analysis (Best Model: {best_name})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "12_per_fault_analysis.png"))
    plt.close()
    print("  Saved: 12_per_fault_analysis.png")


def run_full_evaluation(all_results, signals_dict, X, y, fault_types):
    print(f"\n{'='*60}")
    print("GENERATING EVALUATION FIGURES")
    print(f"{'='*60}\n")

    plot_raw_signals(signals_dict)
    plot_mfcc_spectrograms(signals_dict)
    plot_feature_distributions(X, y, fault_types)
    plot_tsne_clusters(X, y, fault_types)
    plot_training_curves(all_results)
    plot_reconstruction_error_distributions(all_results)
    roc_metrics = plot_roc_curves(all_results)
    plot_precision_recall_curves(all_results)
    plot_confusion_matrices(all_results)
    plot_detection_latency(all_results, X, y, fault_types)
    plot_model_comparison_table(all_results)
    plot_per_fault_analysis(all_results, X, y, fault_types)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    return roc_metrics
