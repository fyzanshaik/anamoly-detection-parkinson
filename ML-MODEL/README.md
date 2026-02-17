# ML Pipeline

Autoencoder-based anomaly detection trained on the CWRU Bearing Dataset with feature-level fusion (statistical + frequency + MFCC).

## Run

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install numpy pandas scipy scikit-learn matplotlib seaborn librosa tensorflow
python src/run_pipeline.py
```

The pipeline downloads data, extracts features, trains 6 models, generates figures, and exports the best model for ESP32. Takes ~45 seconds.

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `download_cwru.py` | Downloads 40 CWRU .mat files (10 fault types x 4 loads) |
| 2 | `features.py` | Extracts 58 features per 2048-sample window |
| 3 | `train.py` | Trains 6 autoencoder configs on normal data only |
| 4 | `evaluate.py` | Generates 12 figures + JSON metrics |
| 5 | `export_esp32.py` | Exports best model as `model_weights.h` |

## Features (58 total)

| Group | Count | Features |
|-------|-------|----------|
| Statistical | 10 | mean, peak, RMS, std, skewness, kurtosis, crest factor, shape factor, impulse factor, clearance factor |
| Frequency | 9 | dominant freq, mean freq, median freq, spectral entropy, total energy, 4 band powers |
| MFCC | 39 | 13 MFCC means + 13 MFCC stds + 13 MFCC deltas |

## Models Trained

| Config | Input | Architecture | Purpose |
|--------|-------|-------------|---------|
| statistical | 10 | 10-6-3-10 | Baseline (time-domain only) |
| frequency | 9 | 9-6-3-9 | Frequency-domain only |
| mfcc | 39 | 39-24-12-39 | MFCC only |
| stat_freq | 19 | 19-12-6-19 | Statistical + frequency fusion |
| fused | 58 | 58-32-16-8-58 | All features fused |
| esp32_deploy | 19 | 19-8-19 | Minimal model for ESP32 (1.4 KB) |

## Outputs

```
outputs/
├── figures/          # 12 publication-quality PNGs
│   ├── 01_raw_signals.png
│   ├── 02_mfcc_spectrograms.png
│   ├── 03_feature_distributions.png
│   ├── 04_tsne_clusters.png
│   ├── 05_training_curves.png
│   ├── 06_reconstruction_errors.png
│   ├── 07_roc_curves.png
│   ├── 08_precision_recall.png
│   ├── 09_confusion_matrices.png
│   ├── 10_detection_latency.png
│   ├── 11_model_comparison.png
│   └── 12_per_fault_analysis.png
├── models/           # Saved Keras models + scaler params
│   ├── model_weights.h   # C++ header for ESP32
│   └── {config}/         # Per-model directory
└── metrics/          # JSON files with all numeric results
```

## Re-running

Delete cached files to force re-computation:
- `data/cwru/cwru_signals.npz` - re-downloads CWRU data
- `data/processed/features.npz` - re-extracts features
- `outputs/models/` - re-trains all models
