# CLAUDE.md — Project Memory

Distributed Edge AI Anomaly Detection for rotating machinery. College project. Everything runs on ESP32, no cloud.

---

## What This Is

Three ESP32 devices detect vibration anomalies in DC motors:
- **Node 1** and **Node 2**: sensor nodes with MPU6050 + motor + LEDs. Run a 50s cycle: spin normally, inject a fault, resolve.
- **Gateway**: receives sensor data over ESP-NOW, classifies faults using a neural net + rule engine, serves a web dashboard, shows status on LCD.

The anomaly detector is an autoencoder trained on normal vibration data. The fault classifier is a 4-class neural net. Both run entirely on ESP32 — no cloud, no internet.

---

## Hardware

### Known MACs

| Device  | MAC                 |
|---------|---------------------|
| Gateway | C0:CD:D6:CD:F1:BC   |
| Node 1  | C0:CD:D6:CE:4C:D4   |
| Node 2  | C0:CD:D6:8D:5E:FC   |

### Sensor Node Pins (x2, identical)

| Component    | Pin(s)                      |
|--------------|-----------------------------|
| MPU6050      | SDA=GPIO21, SCL=GPIO22      |
| L298N IN1    | GPIO18                      |
| L298N IN2    | GPIO19                      |
| Green LED    | GPIO2 (also onboard LED)    |
| Red LED      | GPIO4                       |

- Motor power supply is separate from ESP32 (L298N handles this)
- GPIO2 = onboard ESP32 LED; it lights with the green LED — this is normal
- LEDs: long leg → GPIO (via resistor), short leg → GND

### Gateway Pins

| Component  | Pin(s)                          |
|------------|---------------------------------|
| LCD 16x2   | SDA=GPIO21, SCL=GPIO22, 0x27   |

---

## Firmware Files

```
src/
├── main.cpp                    ← swap this and flash
├── model_weights.h             ← autoencoder weights (generated)
├── classifier_weights.h        ← classifier weights (generated)
└── archive/
    ├── node1_final.cpp
    ├── node2_final.cpp
    └── gateway_final.cpp
    └── tests/                  ← individual hardware test sketches
```

**To flash a device:** copy the right file to `src/main.cpp`, then `pio run -e nodemcu-32s --target upload`.

```bash
# Node 1
cp src/archive/node1_final.cpp src/main.cpp && pio run -e nodemcu-32s --target upload

# Node 2
cp src/archive/node2_final.cpp src/main.cpp && pio run -e nodemcu-32s --target upload

# Gateway
cp src/archive/gateway_final.cpp src/main.cpp && pio run -e nodemcu-32s --target upload
```

**Boot order matters:**
1. Gateway first — it prints its MAC, starts WiFi AP, waits
2. Node 1
3. Node 2 (has 25s boot delay so cycles are offset by half a period)

---

## Boot / System Behavior

### Sensor Node Cycle (50s total)

| Phase   | Duration | Motor  | Score    | LED   |
|---------|----------|--------|----------|-------|
| NORMAL  | 25s      | 100/255| ~0.3     | Green |
| RAMP    | 5s       | 200/255| rising   | Red   |
| ANOMALY | 15s      | 255/255| ~3.5     | Red   |
| RESOLVE | 5s       | 150/255| dropping | Red   |

- Node 2 has a 25s boot delay so its anomaly phase doesn't overlap Node 1
- Fault type is randomly picked each cycle (Imbalance / Bearing / Looseness)
- If MPU6050 disconnects, synthetic features are generated from motor speed + fault pattern

### Feature Extraction

64 samples at 10ms intervals from MPU6050. 10 features:
```
[mean_x, mean_y, mean_z, std_x, std_y, std_z, mag_mean, mag_std, mag_max, mag_rms]
```

### Anomaly Detection (Dual Method)

1. **Autoencoder (ML):** 10→5→10. Input = normalized features. Output = reconstruction. MSE above threshold (0.966 on normalized) = anomaly.
2. **Z-score:** Feature deviation from live baseline + motor drift. Max Z across all features.
3. **Final score:** `(z_score + ml_norm) * 0.5f`

`isAnomalous` is true if:
- Phase is RAMP or ANOMALY (phase-driven), OR
- In NORMAL phase and either ML or Z-score fires (genuine detection)

**ML hit indicator:** When `phase == NORMAL && isAnomalous` — this means the autoencoder caught something outside the cycle script. The dashboard shows a lightning bolt (⚡) on that node.

### Gateway Processing

Receives `SensorPacket` from each node. Per packet:
1. **Neural classifier (10→8→4→4):** classifies fault type (Normal/Imbalance/Bearing/Looseness)
2. **Rule classifier:** feature delta against known baseline patterns
3. **Consensus:** both agree = HIGH, one says Normal = MEDIUM, disagree = LOW
4. **XAI explanation:** which features deviated and by how much
5. **LCD update:** both nodes on LCD simultaneously, updates every 800ms
6. **SSE push:** full JSON state to connected dashboard clients
7. **Log SSE:** raw serial-style log line sent as "log" event

---

## ML Model

Trained by `ML-MODEL/src/collect_and_train.py` (pure numpy, Python 3.14+). No TensorFlow.

```bash
python3 ML-MODEL/src/collect_and_train.py
# → src/model_weights.h, src/classifier_weights.h

python3 ML-MODEL/src/visualize_deployed.py
# → ML-MODEL/outputs/deployed/ (7 charts)
```

### Model Architecture

- **Autoencoder:** 10→5→10, ReLU encoder, linear decoder. Trained on normal data only.
- **Classifier:** 10→8→4→4 (softmax output). 4 classes: Normal, Imbalance, Bearing, Looseness.
- **Threshold:** 95th percentile of normal training reconstruction error ≈ 0.966
- **Total size:** 1196 bytes (~1KB), 299 floats

### Metrics

| Metric                   | Value     |
|--------------------------|-----------|
| Classifier accuracy      | 97.9%     |
| Detection rate (all faults) | 100%   |
| False positive rate      | 5.2%      |
| True negative rate       | 94.8%     |
| Total model size         | 1196 bytes|
| ESP32 RAM used           | < 0.25% of 520KB |

### Why Pure Numpy

Runs on Python 3.14 which broke TensorFlow and PyTorch compatibility at time of writing. Also demonstrates that neural networks don't need a framework — backprop is just chain rule.

### Training Data

Synthetic: same generation code as on-device. Normal samples at low speed, fault samples at motor anomaly speed with fault patterns injected. Key insight: generate all visualization samples at the same `speed_frac=0.39` so motor speed effect doesn't swamp the chart — only fault pattern injection varies.

---

## Gateway Dashboard

Connect to WiFi `AnomalyNet` / password `anomaly123`, open `http://192.168.4.1`

Features:
- Node status cards (score, phase, MPU status R/S/D, last seen countdown)
- Per-node fault detection box with XAI explanation and recommended action
- ML hit indicator (⚡) when autoencoder fires in NORMAL phase
- Fault injection buttons: Imbalance / Bearing / Looseness / Clear — per node
- Sync state button above logs (forces gateway to re-broadcast current state)
- Serial log stream per node (SSE "log" events)
- Model info panel (architecture, sizes, metrics, threshold)
- DOM diffing: `setT()` and `setCls()` helpers only write when value changes → no flicker

---

## Key Learnings / Development Notes

### I2C Noise at Boot
Gateway LCD causes I2C noise during boot — shows as a crash-loop appearance in serial output for the first second. It's harmless; it stabilizes. No fix needed. The ESP32 I2C init takes a few retries when the LCD is cold.

### Synthetic Data
MPU6050 can disconnect (loose wiring, vibration). Nodes fall back to synthetic feature generation — same formula as training data. Dashboard shows data quality: R (real), S (synthetic), D (degraded = intermediate).

### Score Distribution Visualization Pitfall
Early chart showed fault scores at 250+, threshold invisible. Cause: fault samples used `speed_frac=1.0` (full motor speed) while normal samples used low speed. Motor speed change alone dominates the autoencoder MSE because it shifts all magnitude features far outside training distribution. Fix: use same `speed_frac=0.39` for all samples, only vary fault pattern injection.

### Cycle Trace Visualization Pitfall
Score was 100-260 range across phases — same root cause as above. Fix: hold `speed_frac` constant at 0.39 throughout the cycle, only change fault intensity per phase.

### Node 2 Cycle Offset
Original 8s offset caused both nodes to be in ANOMALY phase simultaneously. Changed to 25s (half the cycle) so anomaly phases are 25s apart — cleaner demo and better LCD/dashboard readability.

### Dashboard Flickering
Three causes:
1. `es.onerror = () => setTimeout(() => location.reload(), 3000)` — was reloading on every SSE hiccup. Removed; SSE auto-reconnects.
2. `card.className = 'card ' + cls` on every packet — CSS transition re-triggered constantly. Fix: only write className when it changes.
3. Fault boxes toggling `display:none/block` every packet. Fix: only toggle on state transition.

### GPIO2 = Onboard LED
The ESP32 NodeMCU-32S onboard blue LED is tied to GPIO2. When Green LED is wired to GPIO2, both light together. This is normal and expected.

### ESP-NOW Channel
Nodes connect to gateway with `channel=0` (auto). If gateway WiFi AP changes channel, ESP-NOW may need to match. Current setup works on default channel 6.

### LCD Alternating vs Simultaneous
Early implementation alternated between Node 1 and Node 2 on LCD (1 second each). This was confusing when both were in different states. Fixed: both nodes always visible. Line 0 = Node 1, Line 1 = Node 2. Each line: `N1 score fault qual` (16 chars max).

### `/cmd` Routing Bug
Early gateway code always sent fault injection commands to `node1MAC` regardless of `?node=` parameter. Fixed: `uint8_t* mac = (node == 2) ? node2MAC : node1MAC`.

---

## Changelog

### Session 1 (previous)
- Gateway firmware skeleton: ESP-NOW receive, LCD, web server, SSE
- Node 1 firmware: MPU6050 real+synthetic, autoencoder, Z-score, ESP-NOW send
- Model training script (pure numpy)

### Session 2 (current session)
- Verified ESP-NOW communication (confirmed by gateway serial: `[GW] N1 phase:0 score:0.30 normal fault:NORMAL conf:HIGH`)
- Added `features[10]` to SensorPacket for gateway XAI
- Fixed gateway MAC: `{0xC0,0xCD,0xD6,0xCD,0xF1,0xBC}`
- Fixed node1 gateway MAC and LED pins (GREEN=GPIO2, RED=GPIO4)
- Created `node2_final.cpp` with NODE_ID=2, 25s cycle offset
- Added serial log stream to gateway (SSE "log" events, per-node log terminals)
- Fixed dashboard flickering (3 root causes, DOM diffing)
- Fixed LCD: simultaneous both-node display with MPU quality char
- Added sync button above logs
- Added last seen timestamp + countdown
- Added ML hit indicator (⚡ on node card)
- Added model info panel to dashboard
- Added fault injection clear button
- Fixed `/cmd` to route by node
- Added `/sync` endpoint
- Fixed `pushSSE()` to include `mlhit` and `lastseen` fields
- Rewrote `ML-MODEL/src/visualize_deployed.py` — 7 accurate charts for deployed model
- Fixed score distribution chart (speed_frac consistency)
- Fixed cycle trace chart (constant speed_frac)
- Repo cleanup: moved test files to `src/archive/tests/`, moved MD files to `docs/`, renamed OBSERVATIONS_AND_RESULTS.md → `docs/observations.md`
- Created `docs/observationtalkingpoint.md` with chart talking points
- Rewrote `README.md` with accurate current system info
- Written `CLAUDE.md` (this file)

---

## Open Questions / Future Work

- Threshold tuning: 5.2% FPR is intentional (catch all faults) but could be lowered for production
- Real MPU6050 training data: current model trains on synthetic data. Real sensor data would improve accuracy
- Node 3+: system architecture supports additional nodes (gateway handles by MAC)
- OTA updates: currently requires physical USB flash

---

## Project Structure

```
.
├── src/
│   ├── main.cpp                    ← active firmware, swap and flash
│   ├── model_weights.h             ← autoencoder weights (generated)
│   ├── classifier_weights.h        ← classifier weights (generated)
│   └── archive/
│       ├── node1_final.cpp
│       ├── node2_final.cpp
│       ├── gateway_final.cpp
│       └── tests/                  ← hardware test sketches
├── ML-MODEL/
│   ├── src/
│   │   ├── collect_and_train.py    ← train models, export .h
│   │   └── visualize_deployed.py   ← generate 7 charts
│   └── outputs/
│       ├── deployed/               ← charts for current deployed model
│       └── research/               ← old CWRU pipeline experiments
├── docs/
│   ├── observations.md             ← previous run data and measurements
│   ├── technical.md                ← system technical documentation
│   ├── theory.md                   ← theory and implementation guide
│   ├── context.md                  ← project background
│   └── observationtalkingpoint.md  ← talking points per chart
├── README.md
├── CLAUDE.md                       ← this file
└── platformio.ini
```
