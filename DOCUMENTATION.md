# Technical Documentation

## 1. Problem Statement

Rotating machinery (motors, pumps, compressors) accounts for a significant portion of industrial equipment failures. Traditional maintenance approaches are either reactive (fix after failure, costly downtime) or schedule-based (wasteful, doesn't account for actual condition). This project implements condition-based predictive maintenance using edge AI: vibration sensors on the machinery feed data into a lightweight neural network running directly on a microcontroller, detecting faults in real time without cloud connectivity.

The system addresses three specific fault types common in rotating machinery:
- **Inner race bearing faults** - damage to the inner ring of a ball bearing
- **Outer race bearing faults** - damage to the outer ring
- **Ball faults** - damage to the rolling elements themselves

Each fault type produces a distinct vibration signature that the model learns to distinguish from normal operation.

## 2. Dataset

### CWRU Bearing Dataset

The system is trained and evaluated on the Case Western Reserve University (CWRU) Bearing Dataset, the most widely cited benchmark in bearing fault diagnosis research. The dataset was collected from a 2 HP Reliance Electric motor driving a dynamometer, with accelerometers mounted on the motor housing at the drive end.

**Acquisition setup:**
- Accelerometer: PCB 353B33 (drive end bearing housing)
- Sampling rate: 12,000 Hz
- Motor speeds: 1730-1797 RPM (0-3 HP load)
- Fault diameters: 0.007", 0.014", 0.021" (seeded using electro-discharge machining)

**Dataset composition used in this project:**

| Condition | Fault Sizes | Load Conditions | Files | Total Samples |
|-----------|------------|-----------------|-------|---------------|
| Normal | - | 0, 1, 2, 3 HP | 4 | ~1.7M |
| Inner Race | 7, 14, 21 mil | 0, 1, 2, 3 HP | 12 | ~1.5M |
| Ball | 7, 14, 21 mil | 0, 1, 2, 3 HP | 12 | ~1.5M |
| Outer Race (@6:00) | 7, 14, 21 mil | 0, 1, 2, 3 HP | 12 | ~1.5M |

40 recordings total, each containing 120k-480k acceleration samples (10-40 seconds of data).

### Why CWRU

- Most cited bearing fault dataset in IEEE literature (2000+ papers)
- Real accelerometer data from real motors with real faults
- Multiple fault types, severity levels, and operating conditions
- 12 kHz sampling rate captures the full frequency content relevant to bearing faults
- Standardized benchmark allows direct comparison with published results

## 3. Feature Engineering

Raw vibration signals are segmented into overlapping windows before feature extraction.

**Windowing parameters:**
- Window size: 2048 samples (170.7 ms at 12 kHz)
- Hop size: 1024 samples (85.3 ms) - 50% overlap
- Total windows extracted: 5,886 (1,652 normal + 4,234 anomalous)

Three groups of features are extracted from each window, producing a 58-dimensional feature vector.

### 3.1 Statistical Features (10)

Time-domain features that characterize the amplitude distribution of the vibration signal within each window.

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| Mean | x-bar = (1/N) * sum(x_i) | DC offset / average vibration level |
| Peak | max(\|x_i\|) | Maximum instantaneous amplitude |
| RMS | sqrt((1/N) * sum(x_i^2)) | Signal energy / vibration severity |
| Standard Deviation | sqrt((1/N) * sum((x_i - x-bar)^2)) | Spread of vibration amplitudes |
| Skewness | (1/N) * sum(((x_i - x-bar) / std)^3) | Asymmetry of amplitude distribution |
| Kurtosis | (1/N) * sum(((x_i - x-bar) / std)^4) | Peakedness / impulsive content |
| Crest Factor | Peak / RMS | Ratio of peak to RMS, sensitive to impacts |
| Shape Factor | RMS / mean(\|x\|) | Waveform shape indicator |
| Impulse Factor | Peak / mean(\|x\|) | Sensitivity to impulse-type faults |
| Clearance Factor | Peak / (mean(sqrt(\|x\|)))^2 | Early fault detection indicator |

**Why these features matter:** Bearing faults introduce impulsive components into the vibration signal. Kurtosis and crest factor are particularly sensitive to these impulses, often showing changes before RMS increases. Shape factor and clearance factor are established indicators in ISO 10816 vibration severity standards.

### 3.2 Frequency-Domain Features (9)

Spectral features computed using Welch's power spectral density estimate (256-point FFT segments with Hanning window).

| Feature | What it captures |
|---------|-----------------|
| Dominant Frequency | Frequency with highest power - shifts with fault type |
| Mean Frequency | Centroid of the power spectrum |
| Median Frequency | Frequency dividing spectrum into equal energy halves |
| Spectral Entropy | Flatness/complexity of the spectrum |
| Total Energy | Integral of PSD - overall vibration energy |
| Band Power 0-500 Hz | Low frequency energy (rotor-related faults) |
| Band Power 500-1500 Hz | Mid-low frequency energy |
| Band Power 1500-3000 Hz | Mid-high frequency energy |
| Band Power 3000-6000 Hz | High frequency energy (bearing-related faults) |

**Why these features matter:** Different fault types produce energy at different frequencies. Inner race faults generate characteristic frequencies based on bearing geometry (BPFI), outer race faults at BPFO, and ball faults at BSF. The band power features capture energy redistribution across the spectrum that occurs with different fault types.

### 3.3 MFCC Features (39)

Mel-Frequency Cepstral Coefficients, originally developed for speech recognition, adapted here for vibration analysis. MFCCs capture the spectral envelope of the signal using a perceptually-motivated mel-scale filterbank.

**Extraction process:**
1. Apply 512-point FFT to windowed signal
2. Map power spectrum through 128 mel-scale triangular filters
3. Apply log compression
4. Compute DCT to get 13 cepstral coefficients
5. Aggregate across frames: mean (13), standard deviation (13), delta/first derivative (13)

**Why MFCCs for vibration:** Recent IEEE literature has shown MFCCs capture complementary spectral information that traditional frequency features miss. The mel-scale filterbank provides better resolution at lower frequencies where most mechanical fault signatures appear, while the cepstral representation decorrelates the features, making them more suitable for the autoencoder's reconstruction task.

### 3.4 Feature-Level Fusion

All three feature groups are concatenated into a single 58-dimensional vector per window:

```
[stat_1, ..., stat_10, freq_1, ..., freq_9, mfcc_1, ..., mfcc_39]
```

This approach allows the autoencoder to learn correlations across feature domains. The pipeline trains separate models on each group individually and on the fused set, enabling direct comparison of each group's contribution.

## 4. Model Architecture

### 4.1 Autoencoder for Anomaly Detection

The core model is an undercomplete autoencoder trained exclusively on normal vibration data. The autoencoder learns to compress and reconstruct the feature distribution of healthy machinery. When presented with fault data, the reconstruction error increases because the model has never seen those patterns during training.

```
Input (N features)
    |
    v
[Dense + ReLU] --- Encoder (compression)
    |
    v
Bottleneck (compressed representation)
    |
    v
[Dense + Linear] --- Decoder (reconstruction)
    |
    v
Output (N features, reconstructed)

Anomaly score = MSE(input, output)
If score > threshold --> ANOMALY
```

**Training protocol:**
- Data: Normal samples only (unsupervised anomaly detection)
- Split: 80% train / 20% validation (from normal data only)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Early stopping: patience=15 on validation loss
- Learning rate reduction: factor=0.5, patience=7
- Threshold: 95th percentile of validation reconstruction error

### 4.2 Model Configurations

Six configurations are trained to compare feature groups and architecture sizes:

| Config | Features | Architecture | Parameters | Purpose |
|--------|----------|-------------|------------|---------|
| statistical | 10 stat | 10-6-3-6-10 | 147 | Time-domain baseline |
| frequency | 9 freq | 9-6-3-6-9 | 132 | Frequency-domain baseline |
| mfcc | 39 MFCC | 39-24-12-24-39 | 2,220 | MFCC-only evaluation |
| stat_freq | 19 stat+freq | 19-12-6-12-19 | 534 | Fusion without MFCC |
| fused | 58 all | 58-32-16-8-16-32-58 | 4,538 | Full fusion (all features) |
| esp32_deploy | 19 stat+freq | 19-8-19 | 312 | Edge deployment (minimal) |

The `esp32_deploy` model uses a single hidden layer (19-8-19) to minimize computation on the microcontroller while maintaining detection performance.

### 4.3 Why Autoencoder Over Classification

A supervised classifier (e.g., CNN, SVM) would require labeled fault data for every fault type during training. In practice:
- Not all fault types are known in advance
- Collecting fault data from real machinery is expensive and dangerous
- New fault types should be detected even if never seen during training

The autoencoder approach only needs normal data. Any deviation from normal patterns triggers an alert, making it inherently capable of detecting novel fault types.

## 5. Evaluation Results

### 5.1 Detection Performance

All models achieve near-perfect separation between normal and anomalous samples on the CWRU dataset:

| Model | AUC-ROC | Detection Rate | False Positive Rate |
|-------|---------|----------------|---------------------|
| statistical | 0.9997 | 99.9% | 5.1% |
| frequency | 1.0000 | 100.0% | 5.1% |
| mfcc | 1.0000 | 100.0% | 4.8% |
| stat_freq | 1.0000 | 100.0% | 5.1% |
| fused | 1.0000 | 100.0% | 5.1% |
| esp32_deploy | 1.0000 | 100.0% | 5.1% |

The FPR of ~5% is by design (threshold set at 95th percentile of validation error). This can be tuned: a higher percentile threshold reduces false positives at the cost of slightly delayed detection.

### 5.2 Reconstruction Error Separation

The gap between normal and anomalous reconstruction errors indicates model confidence:

| Model | Normal MSE (mean +/- std) | Anomaly MSE (mean +/- std) | Separation Factor |
|-------|--------------------------|---------------------------|-------------------|
| statistical | 0.202 +/- 0.218 | 689 +/- 1013 | 3,410x |
| frequency | 0.084 +/- 0.109 | 240,469 +/- 113,124 | 2,862,726x |
| esp32_deploy | 0.031 +/- 0.026 | 147,124 +/- 63,500 | 4,746,265x |

The massive separation factors mean the threshold can be set conservatively without missing detections.

### 5.3 Per-Fault Analysis

Using the best model (esp32_deploy), detection rates by fault type:
- Inner race faults (all severities): 100%
- Outer race faults (all severities): 100%
- Ball faults (all severities): 100%

All fault types across all severity levels (7, 14, 21 mil) and all load conditions (0-3 HP) are detected with 100% accuracy.

### 5.4 Detection Latency

Detection latency is the time from fault onset to first detection. With 2048-sample windows at 12 kHz:
- Minimum theoretical latency: 170.7 ms (one window)
- Measured mean latency at optimal threshold: ~85 ms (due to 50% overlap)
- On ESP32 at 500 Hz effective sampling: ~4 seconds per window

The detection latency curves (Figure 10) show the tradeoff between threshold setting and detection speed across all models.

### 5.5 Feature Space Visualization

t-SNE projection of the 58-dimensional fused feature space (Figure 4) shows:
- Normal samples form a tight, distinct cluster
- Each fault type forms its own cluster
- Different severity levels of the same fault type cluster together but remain distinguishable
- Clear separation between normal and all fault types confirms the autoencoder's task is well-defined

## 6. Generated Figures

The pipeline produces 12 publication-ready figures:

| Figure | Content | Use in presentation/paper |
|--------|---------|--------------------------|
| 01_raw_signals.png | Raw vibration waveforms for each fault type | Visual introduction to the data |
| 02_mfcc_spectrograms.png | MFCC coefficient heatmaps per fault | Demonstrates MFCC feature extraction |
| 03_feature_distributions.png | Statistical feature histograms (normal vs anomaly) | Shows feature discriminability |
| 04_tsne_clusters.png | t-SNE 2D projection colored by fault type | Feature space structure |
| 05_training_curves.png | Loss vs epoch for all models | Training convergence |
| 06_reconstruction_errors.png | Error histograms with threshold lines | Anomaly score distributions |
| 07_roc_curves.png | ROC curves for all models | Standard ML evaluation metric |
| 08_precision_recall.png | Precision-recall curves | Performance on imbalanced data |
| 09_confusion_matrices.png | Confusion matrices per model | Classification accuracy breakdown |
| 10_detection_latency.png | Detection rate and latency vs threshold | Real-time performance |
| 11_model_comparison.png | Summary table as figure | Side-by-side comparison |
| 12_per_fault_analysis.png | Error distributions by individual fault type | Fault-specific analysis |

## 7. Edge Deployment

### 7.1 Model Export

The `esp32_deploy` model (19-8-19 architecture) is exported as a C++ header file (`model_weights.h`) containing:
- Encoder weights: 19x8 float matrix (152 floats)
- Encoder bias: 8 floats
- Decoder weights: 8x19 float matrix (152 floats)
- Decoder bias: 19 floats
- Scaler mean: 19 floats (StandardScaler normalization)
- Scaler scale: 19 floats (StandardScaler normalization)
- Anomaly threshold: 1 float

**Total: 369 parameters, 1,476 bytes (1.4 KB)**

This fits comfortably in the ESP32's 520 KB SRAM with negligible memory footprint.

### 7.2 Inference on ESP32

The on-device inference process:

```
1. Collect 2048 accelerometer samples at high rate (buffer)
2. Extract 19 features (10 statistical + 9 frequency)
3. Normalize: x_scaled = (x - scaler_mean) / scaler_scale
4. Forward pass through encoder: hidden = ReLU(x * W_enc + b_enc)
5. Forward pass through decoder: output = hidden * W_dec + b_dec
6. Compute MSE: error = mean((x_scaled - output)^2)
7. Compare: if error > threshold --> ANOMALY
```

Estimated inference time: <1 ms (the bottleneck is data collection, not computation).

### 7.3 Hardware Architecture

**Sensor Node (ESP32 + MPU6050):**
- MPU6050 accelerometer reads 3-axis vibration via I2C
- ESP32 computes magnitude: sqrt(ax^2 + ay^2 + az^2)
- Buffers samples into 2048-sample windows
- Extracts features and runs autoencoder inference
- Sends result via ESP-NOW: {nodeId, isAnomalous, vibrationLevel}
- LED feedback: green=normal, red=anomaly
- DC motor provides the vibration source for demonstration

**Gateway Node (ESP32 + LCD):**
- Receives ESP-NOW messages from sensor nodes
- Displays node status on 16x2 LCD
- Runs XAI explanation on anomaly (fault type classification based on frequency analysis)
- Hosts WiFi access point with web dashboard for monitoring and manual trigger

**Communication:**
- ESP-NOW: Layer 2 protocol, ~288 microsecond transmission time, no TCP/IP overhead
- Message buffering: sensor nodes store up to 20 readings during gateway offline periods
- Heartbeat timeout: 5 seconds to mark a node as offline

## 8. XAI (Explainable AI) on Gateway

When an anomaly is detected, the gateway provides a human-readable explanation of the likely fault type. The explanation pipeline displayed on the LCD:

1. **FFT Analysis** - Compute dominant frequency from the vibration data
2. **Feature Extraction** - Extract the same 8 statistical features
3. **Autoencoder Inference** - Run the model and display reconstruction error
4. **SHAP-style Attribution** - Identify which features contributed most to the high error
5. **Fault Classification** - Map frequency content to known fault patterns:
   - High harmonic ratio + high frequency content --> Bearing Fault
   - Low dominant frequency + high 1x amplitude --> Rotor Imbalance

This XAI pipeline is rule-based on the gateway (not a second ML model) to keep it interpretable and deterministic.

## 9. Reproducing Results

### Prerequisites
- Python 3.11
- uv (package manager)
- PlatformIO (for firmware upload)

### Run ML Pipeline
```bash
cd ML-MODEL
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install numpy pandas scipy scikit-learn matplotlib seaborn librosa tensorflow
python src/run_pipeline.py
```

### Upload Firmware
```bash
# Gateway
cp src/main.cpp.gateway.new src/main.cpp
pio run -t upload

# Sensor (set NODE_ID in code for each node)
cp src/main.cpp.sensor.backup src/main.cpp
pio run -t upload
```

### Output Locations
- Figures: `ML-MODEL/outputs/figures/`
- Trained models: `ML-MODEL/outputs/models/`
- Metrics (JSON): `ML-MODEL/outputs/metrics/`
- ESP32 header: `ML-MODEL/outputs/models/model_weights.h`
