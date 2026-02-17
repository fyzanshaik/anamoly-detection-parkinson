# Distributed Edge AI Anomaly Detection

Predictive maintenance system for rotating machinery using autoencoder neural networks on ESP32 microcontrollers. Trained on the CWRU Bearing Dataset, validated with MFCC feature fusion, and deployed as a 1.4 KB model on edge devices.

## System Architecture

```
Sensor Node 1 (ESP32 + MPU6050 + Motor)
    |
    |--- ESP-NOW (peer-to-peer) ---+
    |                              |
Sensor Node 2 (same hardware)     v
    |                        Gateway Node (ESP32 + LCD)
    |--- ESP-NOW ---------------->  |
                                    |--- WiFi AP --> Web UI (192.168.4.1)
                                    |--- I2C -----> LCD 16x2 Display
```

**Sensor nodes** collect vibration data at high rate, extract statistical and frequency features from windowed samples, run autoencoder inference on-device, and transmit results wirelessly. The **gateway** aggregates status from multiple nodes, runs XAI fault classification, and presents results on an LCD and web dashboard.

## Project Structure

```
.
├── src/                          # ESP32 firmware (PlatformIO / Arduino)
│   ├── main.cpp                  # Active firmware (swap for sensor/gateway)
│   ├── main.cpp.gateway.new      # Gateway firmware
│   └── main.cpp.sensor.backup    # Sensor node firmware
├── ML-MODEL/                     # ML pipeline (Python)
│   ├── src/                      # Pipeline source code
│   │   ├── run_pipeline.py       # Single entry point - runs everything
│   │   ├── download_cwru.py      # CWRU dataset download
│   │   ├── features.py           # Feature extraction (stat + freq + MFCC)
│   │   ├── model.py              # Autoencoder definitions
│   │   ├── train.py              # Training loop (6 model configs)
│   │   ├── evaluate.py           # Metrics + 12 publication figures
│   │   └── export_esp32.py       # Weights --> C++ header for ESP32
│   ├── data/                     # CWRU dataset + processed features
│   └── outputs/                  # Trained models, figures, metrics
├── platformio.ini                # Build config
└── DOCUMENTATION.md              # Full technical documentation
```

## Quick Start

### ML Pipeline

```bash
cd ML-MODEL
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install numpy pandas scipy scikit-learn matplotlib seaborn librosa tensorflow
python src/run_pipeline.py
```

This downloads the CWRU dataset, extracts features, trains 6 autoencoder variants, generates 12 evaluation figures, and exports the best model as a C++ header.

### Firmware Upload

```bash
# Gateway
cp src/main.cpp.gateway.new src/main.cpp
pio run -t upload

# Sensor node (change NODE_ID for each node)
cp src/main.cpp.sensor.backup src/main.cpp
pio run -t upload
```

## Model Performance

Trained on the CWRU Bearing Dataset (12 kHz, 10 fault types, 4 load conditions).

| Model | Features | Dims | AUC-ROC | Detection Rate | FPR | Size |
|-------|----------|------|---------|----------------|-----|------|
| statistical | 10 stat | 10-6-3-10 | 0.9997 | 99.9% | 5.1% | - |
| frequency | 9 freq | 9-6-3-9 | 1.0000 | 100% | 5.1% | - |
| mfcc | 39 MFCC | 39-24-12-39 | 1.0000 | 100% | 4.8% | - |
| stat_freq | 19 stat+freq | 19-12-6-19 | 1.0000 | 100% | 5.1% | - |
| fused | 58 all | 58-32-16-8-58 | 1.0000 | 100% | 5.1% | - |
| **esp32_deploy** | **19 stat+freq** | **19-8-19** | **1.0000** | **100%** | **5.1%** | **1.4 KB** |

## Hardware

| Component | Sensor Node | Gateway |
|-----------|------------|---------|
| MCU | ESP32 NodeMCU-32S | ESP32 NodeMCU-32S |
| Sensor | MPU6050 (I2C) | - |
| Display | - | LCD 16x2 (I2C) |
| Actuator | DC Motor + Driver | - |
| Indicators | Green + Red LEDs | - |
| Communication | ESP-NOW (TX) | ESP-NOW (RX) + WiFi AP |

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full technical writeup covering dataset, feature engineering, model architecture, evaluation methodology, and edge deployment.
