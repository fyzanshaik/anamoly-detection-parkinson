# How It Works — Distributed Edge AI Anomaly Detection

A complete explanation of the system: hardware, execution flow, signal processing, and ML models.

---

## Big Picture

Three ESP32 microcontrollers work together. Two sensor nodes are each attached to a DC motor with a vibration sensor. They continuously measure the motor's vibration, run machine learning on-device to decide if something is wrong, and wirelessly transmit results to a third device — the gateway — which classifies the fault type, displays the result on an LCD screen, and serves a live web dashboard.

Everything runs on hardware that fits in your hand, with no internet connection, no cloud service, and no external server.

```
Sensor Node 1 (ESP32 + MPU6050 + Motor)
        |
        |── ESP-NOW (wireless, no router) ──→
                                              Gateway (ESP32 + LCD)
        |── ESP-NOW ──→                           |── WiFi AP → Web Dashboard
Sensor Node 2 (ESP32 + MPU6050 + Motor)           |── I2C → LCD 16x2
```

---

## Hardware and Why Each Part Was Chosen

### ESP32 (NodeMCU-32S)
The brain of every device. Key reasons for choosing it:
- **240MHz dual-core processor** — fast enough to run neural network inference in microseconds
- **520KB RAM** — orders of magnitude more than a standard Arduino; fits the model weights and buffers comfortably
- **Built-in WiFi and Bluetooth** — the gateway's web dashboard and ESP-NOW wireless communication are both handled by the same chip, no extra hardware needed
- **ESP-NOW protocol** — a Espressif-proprietary peer-to-peer wireless protocol that works at Layer 2 (below IP), meaning no router is required and latency is under 1ms. Sensor nodes connect directly to the gateway by MAC address.
- **Cost** — ~$3–5 per board, making a 3-node distributed system affordable

### MPU6050 (Accelerometer + Gyroscope)
The sensor that "feels" motor vibration.
- 3-axis accelerometer (X, Y, Z) measuring in m/s²
- Communicates via I2C on SDA (GPIO21) and SCL (GPIO22)
- Configured to ±8G range and a 21Hz digital low-pass filter — this smooths high-frequency electrical noise while preserving the mechanical vibration frequencies we care about
- Sampling rate: 100 Hz (one reading every 10ms) — sufficient to capture vibration patterns in motors running at typical speeds

### L298N Motor Driver
Controls the DC motor speed and direction.
- An H-bridge driver — it can supply current in both directions and handle voltages and currents the ESP32 GPIO pins cannot (motors need ~1A, GPIO provides ~12mA)
- IN1 (GPIO18) receives a PWM signal to set speed; IN2 (GPIO19) is held LOW for forward direction
- Motor power comes from a separate supply — the ESP32 and motor share ground but have separate voltage rails

### LCD 16x2 (I2C, address 0x27)
A simple two-line display at the gateway for local status without needing a phone or laptop. I2C means only 2 signal wires (SDA + SCL), which keeps wiring simple.

---

## Sensor Node — Execution Flow

At boot, the node:
1. Initializes serial output, GPIO pins, and motor (stopped)
2. Attempts to connect to MPU6050 over I2C (up to 5 retries)
3. Initializes ESP-NOW and registers the gateway MAC as a peer
4. Starts the motor at low speed (PWM value 100/255)
5. Enters the main loop

### The 50-Second Operating Cycle

Each node runs a repeating cycle designed to demonstrate normal operation followed by a simulated fault condition:

| Phase   | Duration | Motor PWM | Behavior |
|---------|----------|-----------|----------|
| NORMAL  | 25s      | 100/255   | Motor at low speed, stable vibration |
| RAMP    | 5s       | 200/255   | Motor speed increases, fault pattern begins |
| ANOMALY | 15s      | 255/255   | Motor at full speed, fault fully active |
| RESOLVE | 5s       | 150/255   | Motor slows, vibration returns to normal |

The fault type (Imbalance, Bearing, or Looseness) is selected randomly at the start of each RAMP phase using the ESP32's hardware random number generator.

Node 2 waits 25 seconds before starting its cycle — this offsets the two nodes by half a period so they are never in ANOMALY phase simultaneously. This makes the dashboard and LCD clearer, and better demonstrates the system handling two independent machines.

---

## Vibration Measurement and Feature Extraction

This is the core signal processing step — converting raw accelerometer readings into a compact numerical representation the ML model can use.

### Sampling Window

Every loop iteration, the node collects **64 consecutive accelerometer readings** at 10ms intervals. This gives a 640ms time window of vibration data, capturing:
- At least one full rotation at most motor speeds
- Enough samples for statistically meaningful statistics
- Fast enough to detect transient fault signatures like bearing impulses

Each reading is a 3-axis measurement: `(x, y, z)` in m/s².

### Computing the 10 Features

From the 64-sample window, 10 scalar features are computed. Here is exactly what each one means and how it is calculated:

**Feature 0 — mean_x**
Average acceleration in the X axis.
```
mean_x = (1/64) * sum(x_i)
```
Represents the static orientation or mean DC bias of the sensor in X. Shifts when the motor mechanically deflects in one direction.

**Feature 1 — mean_y**
Average acceleration in the Y axis. Same formula. Captures lateral mean displacement.

**Feature 2 — mean_z**
Average acceleration in the Z axis. In a motor mounted vertically, this captures gravitational component plus any vertical mean force.

**Feature 3 — std_x**
Standard deviation of X acceleration across the 64 samples.
```
std_x = sqrt( (1/64) * sum((x_i - mean_x)^2) )
```
Measures how much the motor vibrates along the X axis. A motor running smoothly has low std; a fault increases it.

**Feature 4 — std_y**
Standard deviation of Y acceleration. Same formula. Captures lateral vibration intensity.

**Feature 5 — std_z**
Standard deviation of Z acceleration. Captures vertical vibration intensity.

**Feature 6 — mag_mean**
Mean of the acceleration magnitude across all 64 samples.
```
mag_i = sqrt(x_i^2 + y_i^2 + z_i^2)
mag_mean = (1/64) * sum(mag_i)
```
Magnitude collapses 3 axes into a single "total vibration energy" number. It is direction-agnostic — a fault that increases vibration in any direction raises mag_mean. This is the single most informative feature for detecting anomalies.

**Feature 7 — mag_std**
Standard deviation of the magnitude signal.
```
mag_std = sqrt( (1/64) * sum((mag_i - mag_mean)^2) )
```
Captures how much the total vibration *fluctuates* over the window. A looseness fault creates random bursts of vibration — high mag_std even when mag_mean is moderate.

**Feature 8 — mag_max**
Maximum magnitude value across all 64 samples.
```
mag_max = max(mag_i)
```
Captures peak impulse events. A bearing fault produces brief, sharp impacts — mag_max spikes while mag_mean stays relatively stable. This is the key diagnostic feature for bearing faults.

**Feature 9 — mag_rms**
Root mean square of the magnitude.
```
mag_rms = sqrt( (1/64) * sum(mag_i^2) )
```
RMS is the standard measure of signal power in vibration analysis. It weights large values more than mean does. Similar to mag_mean but more sensitive to peaks.

### Why These Features?

- **Mean values (0-2)** establish the DC offset — faults that cause steady deflection appear here
- **Standard deviations (3-5)** capture vibration intensity per axis — faults increase variance
- **Magnitude features (6-9)** are direction-agnostic and most physically meaningful: real machine faults manifest as changes in total vibration energy, not axis-specific changes
- The magnitude features dominate the model's importance ranking, which validates that the feature engineering is physically correct

---

## Anomaly Detection — Two Methods in Parallel

Two independent methods run on every set of features. Their outputs are combined into a single score.

### Method 1 — Autoencoder (Machine Learning)

An autoencoder is a neural network trained to **reconstruct its own input**. During training, it sees only normal vibration data and learns a compact internal representation of "what normal looks like."

Architecture: **10 → 5 → 10**

```
Input (10 features, normalized)
    ↓
Encoder: linear layer [10×5] + bias + ReLU activation
    ↓
Bottleneck: 5 values (compressed representation)
    ↓
Decoder: linear layer [5×10] + bias (no activation)
    ↓
Output (10 values, reconstructed features)
```

**How it detects anomalies:**

When a normal sample is passed through, the network reconstructs it accurately because it has learned the patterns of normal data. When a fault sample is passed through, the network tries to reconstruct it using its learned normal-data representation — and fails. The difference between input and output is measured as Mean Squared Error (MSE):

```
MSE = (1/10) * sum((input_i - output_i)^2)
```

- Normal input → low MSE (good reconstruction)
- Fault input → high MSE (poor reconstruction)

**Threshold:** Set at the **95th percentile** of MSE values computed on the normal training set. This is not an arbitrary number — it means that 95% of normal readings fall below the threshold, so 5% will be falsely flagged. This is the deliberate design trade-off: accept 5.2% false positives in order to guarantee 100% fault detection rate.

**Normalization:** Before passing features through the autoencoder, each feature is normalized:
```
input_normalized[i] = (feature[i] - mean[i]) / std[i]
```
where `mean` and `std` are computed from the training set. This puts all features on the same scale so no single feature dominates by virtue of having larger absolute values.

**Inference on ESP32:** The weights (encoder and decoder matrices + biases) are baked into a `.h` header file at compile time. Inference is ~18 multiply-accumulate operations per hidden unit — runs in microseconds on the 240MHz CPU.

### Method 2 — Z-Score

A statistical method that compares each feature against a known expected baseline.

For each feature `i`, compute how many standard deviations the current measurement deviates from what is expected at the current motor speed:

```
expected[i] = baseline[i] + MOTOR_DRIFT[i] * speed_fraction
z[i] = |feature[i] - expected[i]| / FEATURE_STD[i]
```

The **motor drift** term accounts for the fact that faster motor speed produces higher vibration. For example, `mag_mean` increases with speed — this is normal physics, not a fault. Without compensating for this, the Z-score would trigger false alarms every time the motor speeds up.

The final Z-score is the **maximum Z across all 10 features** — taking the worst-case deviation.

A Z-score above 1.2 indicates the motor is behaving unusually relative to its expected state at its current speed.

### Combining the Scores

Both scores are normalized to a common range and averaged:

```
ml_norm = (ml_score > threshold) ? 3.0 + ml_score/100 : ml_score/threshold
final_score = (z_score + ml_norm) * 0.5
```

- During normal operation: final_score ≈ 0.3
- During an anomaly: final_score ≈ 3.5

### The `isAnomalous` Decision

The binary anomaly flag is set if any of the following is true:
- Current phase is RAMP or ANOMALY (fault is being actively simulated)
- Current phase is NORMAL **and** either the ML score exceeds its threshold **or** Z-score exceeds 1.2

The second condition is the interesting one — it means the autoencoder or Z-score caught something during a phase when no fault was scheduled. This is a genuine ML detection event, indicated by the ⚡ symbol on the dashboard.

---

## Wireless Communication — ESP-NOW

The sensor node packs its results into a struct and sends it to the gateway over ESP-NOW:

```cpp
struct SensorPacket {
    uint8_t nodeId;        // 1 or 2
    uint8_t phase;         // 0=NORMAL 1=RAMP 2=ANOMALY 3=RESOLVE
    uint8_t dataQuality;   // 0=real 1=synthetic 2=degraded
    bool    isAnomalous;
    float   anomalyScore;
    float   vibrationLevel;
    float   features[10];  // full feature vector
};
```

ESP-NOW works at MAC layer — it does not need an IP address or router. The gateway's MAC address is hardcoded into each node. The packet is sent once per loop iteration (once per ~700ms, the time to collect 64 samples at 10ms each).

The gateway can also send commands back to nodes (fault injection, motor speed override) using a `GatewayCommand` struct over the same ESP-NOW channel.

---

## Gateway — What Happens When a Packet Arrives

The gateway receives a `SensorPacket` from a sensor node and runs through the following pipeline:

### Step 1 — Neural Fault Classifier

Architecture: **10 → 8 → 4 → 4**

```
Input (10 features, normalized)
    ↓
Hidden layer 1: [10×8] + bias + ReLU
    ↓
Hidden layer 2: [8×4] + bias + ReLU
    ↓
Output layer: [4×4] + bias (linear, argmax)
    ↓
Predicted class: 0=Normal, 1=Imbalance, 2=Bearing, 3=Looseness
```

This is a standard feedforward classifier trained with cross-entropy loss and backpropagation using pure numpy — no framework. It maps the 10-feature vector directly to a fault class. Accuracy: 97.9% across 4 classes.

### Step 2 — Rule-Based Classifier

An interpretable, physics-based classifier that uses three derived metrics:

```
impulse_ratio = mag_max / mag_mean
variability   = mag_std / mag_mean
magnitude_delta = mag_mean - baseline_mag_mean
```

Decision logic:
- **Bearing:** impulse_ratio > 1.08 AND mag_max >> mag_mean → sharp spike events relative to average
- **Looseness:** variability > 0.035 AND mag_std elevated → random amplitude variation
- **Imbalance:** magnitude_delta > 0.15 AND std elevated → all axes proportionally elevated
- **Normal:** none of the above

This classifier is fully explainable — every decision traces back to a physical quantity.

### Step 3 — Consensus Engine

The two classifiers vote:
- Both agree → **HIGH** confidence, their result is used
- One says Normal, other says a fault → **MEDIUM** confidence, the fault class is used
- Both say different faults → **LOW** confidence, the neural network result is used

### Step 4 — XAI Explanation

The gateway computes human-readable explanations based on the final fault class and actual feature deltas:

- **Imbalance:** reports how much mag_mean exceeded baseline, describes periodic vibration pattern
- **Bearing:** reports impulse ratio and mag_max delta, notes that mean is relatively stable
- **Looseness:** reports mag_std delta and average std delta across axes, describes irregular pattern
- **Unknown:** flags that pattern is unclear

It also generates a recommended maintenance action specific to the fault type.

### Step 5 — Output

Results are pushed to:
- **LCD:** Both nodes displayed simultaneously on a 16×2 display. Line 0 = Node 1, Line 1 = Node 2. Format: `N1 [score] [fault] [quality]` in 16 characters.
- **Web dashboard:** SSE (Server-Sent Events) stream pushes a JSON packet to all connected browsers. The dashboard updates in real-time without page refresh.
- **Log stream:** A separate SSE event type ("log") pushes a raw serial-style log line per packet received, shown in per-node log terminals on the dashboard.

---

## ML Model Training

The models are trained by `ML-MODEL/src/collect_and_train.py` — a pure numpy implementation with no ML framework dependency.

**Why pure numpy?** The training was done on Python 3.14 at a time when TensorFlow and PyTorch did not yet support it. More fundamentally: a neural network is just matrix multiplications and a chain rule derivative. A framework is not conceptually required.

**Training procedure:**
1. Generate training samples (normal data for autoencoder; all 4 classes for classifier)
2. Normalize features (zero mean, unit variance)
3. Train autoencoder with MSE loss using mini-batch gradient descent, 500 epochs
4. Set threshold at 95th percentile of normal reconstruction errors
5. Train classifier with cross-entropy loss + softmax output, 400 epochs
6. Export all weights and normalization parameters as C float arrays to `.h` header files

The exported header files (`model_weights.h`, `classifier_weights.h`) are compiled directly into the ESP32 firmware. The model lives in flash memory and is loaded into RAM at runtime for inference.

**Total model size: 1196 bytes.** The ESP32 has 520KB of RAM — the models use less than 0.25% of available memory.

---

## End-to-End Data Flow Summary

```
MPU6050 sensor
    ↓ raw x,y,z samples (64 × 10ms)
Feature extraction (10 statistics)
    ↓ [mean_x, mean_y, mean_z, std_x, std_y, std_z, mag_mean, mag_std, mag_max, mag_rms]
Dual anomaly detection
    ↓ Autoencoder MSE + Z-score → combined anomaly score + isAnomalous flag
SensorPacket assembly
    ↓ nodeId, phase, quality, isAnomalous, score, vibration, features[10]
ESP-NOW wireless transmission (MAC layer, no router)
    ↓
Gateway receives packet
    ↓
Neural classifier (10→8→4→4) + Rule classifier
    ↓
Consensus engine → fault type + confidence
    ↓
XAI explanation generator
    ↓
LCD update (both nodes, 16×2, I2C)
    ↓
SSE push → web dashboard (live, no refresh)
```

---

## Web Dashboard

Connect to WiFi: `AnomalyNet` / `anomaly123` → open `http://192.168.4.1`

The dashboard receives data via Server-Sent Events — a one-way persistent HTTP stream from the gateway to the browser. No polling, no WebSocket. The browser keeps a connection open and the gateway pushes updates as they arrive (~every 700ms per node).

DOM diffing prevents flickering: JavaScript helper functions `setT()` and `setCls()` check the current DOM value before writing. If the value hasn't changed, no write happens — CSS transitions don't re-trigger and the page doesn't flash.
