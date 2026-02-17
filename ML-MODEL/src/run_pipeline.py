#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from download_cwru import download_and_prepare, DATA_DIR, SAMPLING_RATE
from features import extract_dataset_features, ALL_FEATURE_NAMES
from train import train_all_configs
from evaluate import run_full_evaluation
from export_esp32 import export_for_esp32

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def step_1_download():
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD CWRU BEARING DATASET")
    print("=" * 70)

    signals_path = os.path.join(DATA_DIR, "cwru_signals.npz")
    meta_path = os.path.join(DATA_DIR, "cwru_metadata.npy")

    if os.path.exists(signals_path) and os.path.exists(meta_path):
        print("Dataset already downloaded, loading from cache...")
        npz = np.load(signals_path)
        meta = np.load(meta_path, allow_pickle=True).item()
        signals = {}
        for key in npz.files:
            signals[key] = {
                "signal": npz[key],
                **meta[key]
            }
        print(f"  Loaded {len(signals)} signals")
    else:
        signals = download_and_prepare()

    return signals


def step_2_extract_features(signals):
    print("\n" + "=" * 70)
    print("STEP 2: EXTRACT FEATURES (Statistical + Frequency + MFCC)")
    print("=" * 70)

    processed_dir = os.path.join(os.path.dirname(DATA_DIR), "processed")
    os.makedirs(processed_dir, exist_ok=True)

    cache_path = os.path.join(processed_dir, "features.npz")
    if os.path.exists(cache_path):
        print("Features already extracted, loading from cache...")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        fault_types = data["fault_types"]
        loads = data["loads"]
        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        print(f"Window size: 2048 samples ({2048/SAMPLING_RATE*1000:.0f}ms)")
        print(f"Hop size: 1024 samples ({1024/SAMPLING_RATE*1000:.0f}ms)")
        print(f"Features per window: {len(ALL_FEATURE_NAMES)}")
        print()

        X, y, fault_types, loads, keys = extract_dataset_features(signals, SAMPLING_RATE)

        np.savez_compressed(cache_path, X=X, y=y, fault_types=fault_types, loads=loads, keys=keys)
        print(f"\n  Saved features to: {cache_path}")

    nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        print(f"  Removing {n_bad} samples with NaN/Inf values")
        X = X[~nan_mask]
        y = y[~nan_mask]
        fault_types = fault_types[~nan_mask]
        loads = loads[~nan_mask]

    print(f"\n  Total samples: {len(X)}")
    print(f"  Normal: {np.sum(y == 0)}, Anomaly: {np.sum(y == 1)}")
    print(f"  Features: {X.shape[1]} ({len(ALL_FEATURE_NAMES)} per window)")

    unique_faults, counts = np.unique(fault_types, return_counts=True)
    for ft, cnt in zip(unique_faults, counts):
        print(f"    {ft}: {cnt}")

    return X, y, fault_types, loads


def step_3_train(X, y, fault_types):
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN AUTOENCODER MODELS")
    print("=" * 70)

    configs_to_train = ["statistical", "frequency", "mfcc", "stat_freq", "fused", "esp32_deploy"]

    all_results = train_all_configs(X, y, fault_types, configs=configs_to_train)
    return all_results


def step_4_evaluate(all_results, signals, X, y, fault_types):
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATE AND GENERATE FIGURES")
    print("=" * 70)

    roc_metrics = run_full_evaluation(all_results, signals, X, y, fault_types)
    return roc_metrics


def step_5_export():
    print("\n" + "=" * 70)
    print("STEP 5: EXPORT BEST MODEL FOR ESP32")
    print("=" * 70)

    header_path = export_for_esp32("esp32_deploy")
    return header_path


def main():
    start_time = time.time()

    print("=" * 70)
    print("  EDGE AI ANOMALY DETECTION - ML PIPELINE")
    print("  CWRU Bearing Dataset + Feature Fusion + Autoencoder")
    print("=" * 70)

    signals = step_1_download()
    X, y, fault_types, loads = step_2_extract_features(signals)
    all_results = step_3_train(X, y, fault_types)
    roc_metrics = step_4_evaluate(all_results, signals, X, y, fault_types)
    header_path = step_5_export()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"\n  Outputs:")
    print(f"    Models:  {os.path.join(OUTPUT_DIR, 'models')}/")
    print(f"    Figures: {os.path.join(OUTPUT_DIR, 'figures')}/")
    print(f"    Metrics: {os.path.join(OUTPUT_DIR, 'metrics')}/")
    if header_path:
        print(f"    ESP32:   {header_path}")
    print()

    print("  Model Performance Summary:")
    for name, res in all_results.items():
        r = res["results"]
        auc_val = roc_metrics.get(name, {}).get("auc_roc", 0)
        print(f"    {name:<15} AUC={auc_val:.4f}  Detect={r['detection_rate']:.1f}%  FPR={r['false_positive_rate']:.1f}%")

    print()


if __name__ == "__main__":
    main()
