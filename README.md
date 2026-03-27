# Distributed Edge AI Anomaly Detection

Predictive maintenance system for rotating machinery. Two ESP32 sensor nodes with MPU6050 accelerometers and DC motors detect vibration anomalies using an autoencoder neural network running entirely on-device. A gateway ESP32 receives data wirelessly, classifies the fault type using a neural classifier + rule engine, and presents results on an LCD display and web dashboard.

No cloud. No internet. Fully self-contained edge AI.

---

## System Architecture

```
Sensor Node 1 (ESP32 + MPU6050 + Motor + LEDs)
    |
    |--- ESP-NOW (peer-to-peer wireless) ---+
    |                                       |
Sensor Node 2 (same hardware)              v
    |                               Gateway Node (ESP32 + LCD 16x2)
    |--- ESP-NOW ----------------------->   |
                                            |--- WiFi AP (AnomalyNet) --> Web Dashboard (192.168.4.1)
                                            |--- I2C --> LCD 16x2 (0x27)
```

---

## Hardware

### Sensor Node (x2) — identical hardware

| Component | Model | Pin |
|-----------|-------|-----|
| MCU | ESP32 NodeMCU-32S | — |
| Accelerometer | MPU6050 | SDA=GPIO21, SCL=GPIO22 |
| Motor Driver | L298N | IN1=GPIO18, IN2=GPIO19 |
| DC Motor | TT Gear Motor | Via L298N |
| Green LED | 5mm | GPIO2 |
| Red LED | 5mm | GPIO4 |

**Wiring notes:**
- MPU6050: VCC → 3.3V, GND → GND, SDA → GPIO21, SCL → GPIO22
- L298N: IN1 → GPIO18, IN2 → GPIO19, GND → common GND, motor power separate supply
- LEDs: long leg → GPIO pin (through resistor), short leg → GND
- GPIO2 is also the ESP32 onboard LED — it will light with the green LED, this is normal

### Gateway Node (x1)

| Component | Model | Pin |
|-----------|-------|-----|
| MCU | ESP32 NodeMCU-32S | — |
| LCD | 16x2 I2C | SDA=GPIO21, SCL=GPIO22, addr=0x27 |

---

## Firmware

All firmware is managed via PlatformIO. `src/main.cpp` is what gets compiled and flashed. Swap the file to target a different device.

### Flashing Node 1
```bash
cp src/archive/node1_final.cpp src/main.cpp
pio run -e nodemcu-32s --target upload
```

### Flashing Node 2
```bash
cp src/archive/node2_final.cpp src/main.cpp
pio run -e nodemcu-32s --target upload
```

### Flashing Gateway
```bash
cp src/archive/gateway_final.cpp src/main.cpp
pio run -e nodemcu-32s --target upload
```

### Monitor serial output
```bash
pio device monitor --baud 115200
```

---

## Boot Order

Always boot in this order to avoid ESP-NOW pairing issues:

1. **Gateway** — boots, prints MAC address, starts WiFi AP and waits
2. **Node 1** — connects to gateway MAC, starts 50s cycle
3. **Node 2** — connects to gateway MAC, starts 50s cycle with 25s offset

---

## How It Works

### Sensor Node

Each sensor node runs a 50-second auto-cycle:

| Phase | Duration | Motor | What happens |
|-------|----------|-------|-------------|
| NORMAL | 25s | Low (100/255) | Score ~0.3, green LED |
| RAMP | 5s | Medium (200/255) | Score rises, fault pattern begins |
| ANOMALY | 15s | Full (255/255) | Score ~3.5, red LED, ANOMALY sent |
| RESOLVE | 5s | Medium-low (150/255) | Score drops back |

**Feature extraction:** 64 accelerometer samples are collected at 10ms intervals. 10 features are computed: mean_x/y/z, std_x/y/z, mag_mean, mag_std, mag_max, mag_rms.

**Anomaly detection:** Two methods run in parallel:
- **Autoencoder (ML):** Features normalized and passed through a 10→5→10 autoencoder trained on normal data. Reconstruction error above threshold = anomaly.
- **Z-score:** Features compared against expected baseline + motor drift. Max deviation across all features.

**Synthetic data:** If MPU6050 is disconnected, the node generates synthetic features based on motor speed and fault pattern. This ensures the demo works even with loose wiring.

**Packet sent to gateway:**
```cpp
struct SensorPacket {
    uint8_t  nodeId;          // 1 or 2
    uint8_t  phase;           // 0=NORMAL 1=RAMP 2=ANOMALY 3=RESOLVE
    uint8_t  dataQuality;     // 0=real MPU 1=synthetic 2=degraded
    bool     isAnomalous;
    float    anomalyScore;
    float    vibrationLevel;
    float    features[10];    // full feature vector for gateway XAI
};
```

### Gateway

Receives SensorPacket from each node via ESP-NOW and runs:

1. **Neural classifier (10→8→4→4):** Classifies fault type — Normal, Imbalance, Bearing, Looseness
2. **Rule classifier:** Feature delta analysis against known baseline patterns
3. **Consensus engine:** Both agree = HIGH confidence, one says Normal = MEDIUM, disagree = LOW
4. **XAI explanation:** Per-fault explanation with actual feature delta values

Results pushed to LCD and web dashboard via SSE.

### Web Dashboard

Connect to WiFi `AnomalyNet` / `anomaly123`, open `192.168.4.1`.

- Live node status cards (score, phase, MPU status)
- Per-node fault detection box with XAI explanation and recommended action
- Fault injection buttons (N1/N2 Imbalance / Bearing / Looseness / Clear)
- Sync state button — forces gateway to re-broadcast current state
- Serial log stream per node
- Model info panel

**ML hit indicator:** When the autoencoder fires during NORMAL phase (not phase-driven), the dashboard shows a lightning bolt indicator on that node card.

---

## ML Model

Trained by `ML-MODEL/src/collect_and_train.py` using pure numpy (no TensorFlow/PyTorch). Runs on Python 3.14+.

```bash
python3 ML-MODEL/src/collect_and_train.py
# Outputs: src/model_weights.h, src/classifier_weights.h

python3 ML-MODEL/src/visualize_deployed.py
# Outputs: ML-MODEL/outputs/deployed/ (7 charts)
```

| Metric | Value |
|--------|-------|
| Autoencoder | 10→5→10, threshold=0.966 |
| Classifier | 10→8→4→4, 97.9% accuracy |
| Total model size | 1196 bytes (~1KB) |
| Parameters | 299 floats |
| Detection rate | 100% (all fault types) |
| False positive rate | 5.2% |

---

## Project Structure

```
.
├── src/
│   ├── main.cpp                    # Active firmware — swap and flash
│   ├── model_weights.h             # Autoencoder weights (generated)
│   ├── classifier_weights.h        # Classifier weights (generated)
│   └── archive/
│       ├── node1_final.cpp         # Node 1 firmware
│       ├── node2_final.cpp         # Node 2 firmware
│       └── gateway_final.cpp       # Gateway firmware
│       └── tests/                  # Individual hardware test sketches
├── ML-MODEL/
│   ├── src/
│   │   ├── collect_and_train.py    # Train both models, export .h files
│   │   └── visualize_deployed.py   # Generate 7 evaluation charts
│   └── outputs/
│       ├── deployed/               # Charts for current deployed model
│       └── research/               # Charts from CWRU pipeline experiments
├── docs/
│   ├── observations.md             # Previous run data and measurements
│   ├── technical.md                # System technical documentation
│   ├── theory.md                   # Implementation and theory guide
│   ├── context.md                  # Project background
│   └── observationtalkingpoint.md  # Presentation talking points per chart
└── platformio.ini
```

---

## Known MACs

| Device | MAC |
|--------|-----|
| Gateway | C0:CD:D6:CD:F1:BC |
| Node 1 | C0:CD:D6:CE:4C:D4 |
| Node 2 | C0:CD:D6:8D:5E:FC |
