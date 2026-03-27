# Comprehensive Theory and Implementation Guide

## Distributed Anomaly Detection System with Edge ML

---

## PART 1: THEORETICAL FOUNDATIONS

### 1.1 Edge Computing Fundamentals

**What is Edge Computing?** Edge computing processes data near its source rather than sending it to centralized servers. In our system, each ESP32 acts as an edge node, making decisions locally
instead of relying on cloud processing.

**Why Edge Over Cloud:**

-  **Latency**: Local processing = millisecond response vs seconds for cloud
-  **Bandwidth**: Only send results, not raw data (99% reduction)
-  **Reliability**: Works without internet connection
-  **Privacy**: Sensitive vibration data stays local

### 1.2 Vibration Analysis Theory

**Understanding Vibration Signatures:** Every rotating machine produces a characteristic vibration pattern determined by:

-  Rotational speed (RPM)
-  Number of components (bearings, blades)
-  Mechanical condition

**Common Fault Patterns:**

```
Normal Operation:
├── Steady amplitude
├── Dominant frequency at 1× RPM
└── Low noise floor

Imbalance:
├── High amplitude at 1× RPM
├── Phase shift between measurements
└── Increases with speed squared

Misalignment:
├── High amplitude at 2× RPM
├── Axial vibration present
└── Multiple harmonics

Bearing Failure:
├── High frequency components (>10× RPM)
├── Random noise increases
└── Non-harmonic peaks
```

### 1.3 Signal Processing Essentials

**Raw Data to Features:** The MPU6050 provides acceleration in X, Y, Z axes. We convert this to useful metrics:

**1. Magnitude Calculation:**

```
Total_Acceleration = √(X² + Y² + Z²)
```

This gives direction-independent vibration strength.

**2. RMS (Root Mean Square):**

```
RMS = √(1/N × Σ(xi²))
```

Represents average vibration energy over time.

**3. Peak-to-Peak:**

```
P2P = Max_value - Min_value
```

Captures vibration range.

**4. Frequency Domain (if implementing FFT):**

-  Converts time-series data to frequency spectrum
-  Reveals periodic patterns invisible in time domain
-  Critical for identifying specific fault types

---

## PART 2: MACHINE LEARNING IMPLEMENTATION

### 2.1 Approach Options by Complexity

#### Option A: Statistical Threshold (No ML) - 2 Hours

**Implementation:**

```cpp
// Simple threshold detection
float vibrationMagnitude = sqrt(ax*ax + ay*ay + az*az);
bool isAnomalous = (vibrationMagnitude > THRESHOLD);
```

**Threshold Calibration:**

1. Measure normal operation for 60 seconds
2. Calculate mean + 3×standard_deviation
3. Set as threshold

**Pros:** Simple, deterministic, explainable **Cons:** Can't adapt, many false positives

#### Option B: Adaptive Thresholds - 4 Hours

**Implementation:**

```cpp
// Moving average with adaptive threshold
class AdaptiveDetector {
    float movingAverage;
    float movingStdDev;
    const float ALPHA = 0.1;  // Smoothing factor

    void update(float value) {
        movingAverage = ALPHA * value + (1-ALPHA) * movingAverage;
        float deviation = abs(value - movingAverage);
        movingStdDev = ALPHA * deviation + (1-ALPHA) * movingStdDev;
    }

    bool detectAnomaly(float value) {
        return abs(value - movingAverage) > 3 * movingStdDev;
    }
};
```

**Pros:** Adapts to changing conditions **Cons:** Still rule-based, not true ML

#### Option C: Lightweight Autoencoder - 8+ Hours

**Full ML Pipeline:**

**1. Data Collection Phase:**

```python
# Collect training data (on PC)
import serial
import pandas as pd

data = []
with serial.Serial('COM3', 115200) as ser:
    for i in range(10000):  # Collect 10k samples
        line = ser.readline().decode()
        x, y, z = map(float, line.split(','))
        data.append([x, y, z])

df = pd.DataFrame(data, columns=['x', 'y', 'z'])
df.to_csv('normal_operation.csv')
```

**2. Feature Engineering:**

```python
# Create features from raw data
import numpy as np

def extract_features(window):
    features = []
    features.append(np.mean(window, axis=0))  # Mean
    features.append(np.std(window, axis=0))   # Std deviation
    features.append(np.max(window, axis=0))   # Peak
    features.append(np.min(window, axis=0))   # Valley

    # Frequency domain (optional)
    fft = np.fft.fft(window, axis=0)
    features.append(np.abs(fft[:10]))  # First 10 frequencies

    return np.concatenate(features)
```

**3. Model Training:**

```python
from tensorflow import keras
import tensorflow as tf

# Simple autoencoder
input_dim = 12  # Number of features
encoding_dim = 4  # Compression

input_layer = keras.Input(shape=(input_dim,))
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal data only
autoencoder.fit(X_normal, X_normal, epochs=50, batch_size=32)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**4. On-Device Inference:**

```cpp
// ESP32 code for TFLite inference
#include <TensorFlowLite_ESP32.h>

// Model as byte array
const unsigned char model_data[] = { /* converted model */ };

// Inference
float input_data[12];  // Features
float output_data[12]; // Reconstruction

// Fill input_data with features...

interpreter->Invoke();

// Calculate reconstruction error
float error = 0;
for(int i = 0; i < 12; i++) {
    error += abs(input_data[i] - output_data[i]);
}

bool isAnomalous = (error > THRESHOLD);
```

### 2.2 Choosing Your Approach

**Decision Tree:**

```
Time Available?
├── < 4 hours → Use Statistical Thresholds
├── 4-6 hours → Implement Adaptive Thresholds
└── > 8 hours → Try Lightweight ML
    ├── Have ML experience? → Full Autoencoder
    └── New to ML? → Simplified feature-based model
```

---

## PART 3: COMMUNICATION PROTOCOL DEEP DIVE

### 3.1 ESP-NOW Protocol Internals

**How ESP-NOW Works:**

-  Operates at Data Link Layer (Layer 2)
-  Bypasses TCP/IP stack = ultra-low latency
-  Uses Action Frames from 802.11 standard
-  Maximum payload: 250 bytes
-  Encryption: CCMP (optional)

**Message Structure:**

```cpp
// Optimize for efficiency
struct __attribute__((packed)) SensorMessage {
    uint8_t nodeId;        // 1 byte
    uint8_t status;        // 1 byte (0=normal, 1=warning, 2=critical)
    float vibration;       // 4 bytes
    uint16_t battery_mV;   // 2 bytes
    uint32_t timestamp;    // 4 bytes
};  // Total: 12 bytes
```

**Timing Considerations:**

```
Transmission time = Tpreamble + Theader + Tdata + Tack
                  = 192μs + 40μs + (12×8/6)μs + 40μs
                  = ~288μs per message
```

### 3.2 Reliable Communication Patterns

**Implement Acknowledgment System:**

```cpp
// Sensor node code
volatile bool msgSent = false;
volatile bool msgFailed = false;

void OnDataSent(const uint8_t *mac, esp_now_send_status_t status) {
    msgSent = true;
    msgFailed = (status != ESP_NOW_SEND_SUCCESS);
}

void sendWithRetry(SensorMessage &msg, int maxRetries = 3) {
    for(int i = 0; i < maxRetries; i++) {
        msgSent = false;
        esp_now_send(gatewayMAC, (uint8_t*)&msg, sizeof(msg));

        // Wait for callback
        unsigned long start = millis();
        while(!msgSent && millis() - start < 100) {
            delay(1);
        }

        if(!msgFailed) break;  // Success
        delay(50 * i);  // Exponential backoff
    }
}
```

---

## PART 4: DEBUGGING STRATEGIES

### 4.1 Systematic Debugging Approach

**Layer-by-Layer Testing:**

**1. Hardware Layer:**

```cpp
// Test each component individually
void testHardware() {
    // Test I2C
    Wire.beginTransmission(0x68);  // MPU6050 address
    if(Wire.endTransmission() == 0) {
        Serial.println("MPU6050 found");
    }

    // Test motor
    digitalWrite(MOTOR_PIN, HIGH);
    delay(1000);
    digitalWrite(MOTOR_PIN, LOW);

    // Test LEDs
    digitalWrite(GREEN_LED, HIGH);
    delay(500);
    digitalWrite(RED_LED, HIGH);
}
```

**2. Sensor Layer:**

```cpp
// Visualize sensor data
void debugSensor() {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    Serial.print("Accel X:");
    Serial.print(a.acceleration.x);
    Serial.print(" Y:");
    Serial.print(a.acceleration.y);
    Serial.print(" Z:");
    Serial.println(a.acceleration.z);

    // Check for sensor saturation
    if(abs(a.acceleration.x) > 15) {
        Serial.println("WARNING: Sensor saturated!");
    }
}
```

**3. Communication Layer:**

```cpp
// ESP-NOW debugging
void debugComm() {
    // Print MAC address
    Serial.print("MAC: ");
    Serial.println(WiFi.macAddress());

    // Monitor packet loss
    static int sent = 0, failed = 0;
    sent++;
    if(msgFailed) failed++;

    Serial.printf("Packet Loss: %.1f%%\n",
                  100.0 * failed / sent);
}
```

### 4.2 Common Issues and Solutions

**Issue: Noisy Sensor Readings**

```cpp
// Solution: Implement filtering
class LowPassFilter {
    float alpha = 0.8;  // Smoothing factor
    float filtered = 0;

public:
    float update(float raw) {
        filtered = alpha * filtered + (1 - alpha) * raw;
        return filtered;
    }
};
```

**Issue: I2C Communication Fails**

```cpp
// Solution: I2C scanner
void scanI2C() {
    for(byte addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if(Wire.endTransmission() == 0) {
            Serial.printf("Device at 0x%02X\n", addr);
        }
    }
}
```

**Issue: Intermittent ESP-NOW Failures**

```cpp
// Solution: Channel and power management
void setupReliableComm() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();  // Clear any saved networks

    // Set specific channel
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

    // Reduce TX power if nodes are close
    esp_wifi_set_max_tx_power(50);  // 12.5dBm
}
```

---

## PART 5: PERFORMANCE OPTIMIZATION

### 5.1 Memory Management

**ESP32 Memory Layout:**

```
DRAM: 520KB total
├── Static data: ~50KB
├── Stack: 8KB per task
├── Heap: Remaining (~400KB available)
└── DMA buffers: As needed

IRAM: 128KB (instructions)
Flash: 4MB (code + constants)
```

**Optimization Strategies:**

```cpp
// Use PROGMEM for constants
const char messages[] PROGMEM = "Status messages...";

// Minimize dynamic allocation
// Bad:
String message = "Node " + String(nodeId);

// Good:
char message[32];
snprintf(message, 32, "Node %d", nodeId);
```

### 5.2 Power Optimization

**For Battery-Powered Nodes:**

```cpp
void enterLightSleep(int seconds) {
    esp_sleep_enable_timer_wakeup(seconds * 1000000);
    esp_light_sleep_start();
}

void loop() {
    // Take measurement
    float vibration = readSensor();

    // Send if anomalous or every minute
    if(isAnomalous(vibration) || millis() - lastSend > 60000) {
        sendStatus();
        lastSend = millis();
    }

    // Sleep between measurements
    enterLightSleep(1);  // Sleep 1 second
}
```

---

## PART 6: EXPLAINABLE AI IMPLEMENTATION

### 6.1 Simple Rule-Based Explanations

**For Quick Implementation:**

```cpp
String generateExplanation(float vibration, float baseline) {
    float ratio = vibration / baseline;

    if(ratio > 3.0) {
        return "CRITICAL: Severe vibration - Check mounting";
    } else if(ratio > 2.0) {
        if(isPeriodicPattern()) {
            return "Imbalance detected - Check rotor";
        } else {
            return "Bearing wear suspected";
        }
    } else if(ratio > 1.5) {
        return "Minor anomaly - Monitor closely";
    }
    return "Normal operation";
}
```

### 6.2 Feature Attribution (Advanced)

**If implementing ML:**

```cpp
// Track feature importance
struct FeatureImportance {
    float weights[12];  // For 12 features

    String explainAnomaly(float features[]) {
        // Find most deviant features
        int maxIndex = 0;
        float maxDeviation = 0;

        for(int i = 0; i < 12; i++) {
            float deviation = abs(features[i] - normalProfile[i]);
            deviation *= weights[i];  // Weight by importance

            if(deviation > maxDeviation) {
                maxDeviation = deviation;
                maxIndex = i;
            }
        }

        // Map feature to explanation
        switch(maxIndex) {
            case 0: return "High X-axis vibration";
            case 1: return "High Y-axis vibration";
            case 2: return "High Z-axis vibration";
            case 3: return "Increased variance";
            // ... etc
        }
    }
};
```

---

## PART 7: QUICK IMPLEMENTATION CHECKLIST

### Minimum Viable Product (4 hours):

-  [ ] ESP-NOW communication working
-  [ ] Basic sensor reading (raw values)
-  [ ] Simple threshold detection
-  [ ] LED status indicators
-  [ ] Gateway receives messages
-  [ ] Basic OLED display

### Enhanced Version (+4 hours):

-  [ ] Filtered sensor data
-  [ ] Adaptive thresholds
-  [ ] Proper vibration magnitude calculation
-  [ ] Retry logic for communication
-  [ ] Detailed OLED explanations
-  [ ] Multiple fault patterns

### ML Version (+8 hours):

-  [ ] Data collection script
-  [ ] Feature extraction
-  [ ] Model training pipeline
-  [ ] TFLite conversion
-  [ ] On-device inference
-  [ ] Performance metrics

This comprehensive guide provides all the theory and practical details needed to implement the system at any complexity level, from basic to full ML implementation.
# Distributed Anomaly Detection System for Predictive Maintenance

## Complete Implementation Guide

---

## 1. PROJECT OVERVIEW

### What It Is

A distributed edge computing system that monitors multiple motors in real-time, detects anomalies locally using embedded AI, and provides centralized explanations for maintenance decisions.

### What It Does

-  **Monitors** vibration patterns of 3 motors simultaneously
-  **Detects** anomalies at each sensor node independently (edge processing)
-  **Communicates** status wirelessly using ESP-NOW protocol
-  **Explains** the type and cause of detected anomalies at a central gateway
-  **Visualizes** system health through LEDs and OLED display

### Real-World Application

Simulates an industrial predictive maintenance system where early fault detection prevents costly machine failures through continuous monitoring and intelligent analysis.

---

## 2. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                    DISTRIBUTED NETWORK                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  SENSOR NODE 1          SENSOR NODE 2      SENSOR NODE 3│
│  ┌──────────┐           ┌──────────┐       ┌──────────┐│
│  │ ESP32 #1 │           │ ESP32 #2 │       │ ESP32 #3 ││
│  │   +      │           │   +      │       │   +      ││
│  │ MPU6050  │           │ MPU6050  │       │ MPU6050  ││
│  │   +      │           │   +      │       │   +      ││
│  │ Motor 1  │           │ Motor 2  │       │ Motor 3  ││
│  └────┬─────┘           └────┬─────┘       └────┬─────┘│
│       │                      │                   │      │
│       └──────────┬───────────┴───────────┬──────┘      │
│                  ↓                       ↓              │
│                     ESP-NOW Wireless                    │
│                          (2.4 GHz)                      │
│                              ↓                          │
│                    ┌─────────────────┐                  │
│                    │  GATEWAY NODE   │                  │
│                    │    ESP32 #4     │                  │
│                    │       +         │                  │
│                    │  OLED Display   │                  │
│                    └─────────────────┘                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. COMPONENT FUNCTIONS

### Sensor Nodes (3 units)

**Components per node:**

-  ESP32: Microcontroller brain
-  MPU6050: 6-axis accelerometer/gyroscope for vibration detection
-  DC Motor: Simulates industrial machinery
-  LEDs: Visual status indicators (Green=Normal, Red=Anomaly)

**Functions:**

1. Continuously reads vibration data from MPU6050
2. Processes data locally to detect anomalies
3. Sends status updates to gateway via ESP-NOW
4. Controls status LED based on detection results

### Gateway Node (1 unit)

**Components:**

-  ESP32: Central processing unit
-  OLED Display: Shows system status and explanations

**Functions:**

1. Receives status messages from all sensor nodes
2. Aggregates system health information
3. Generates explanations for detected anomalies
4. Displays results on OLED screen

---

## 4. CONNECTION DIAGRAMS

### Sensor Node Connections (Repeat for each of 3 nodes)

```
ESP32 SENSOR NODE:
┌──────────────┐
│     ESP32    │
│              │
│ 3V3 ────────────→ MPU6050 VCC
│ GND ────────────→ MPU6050 GND
│ GPIO21 (SDA)────→ MPU6050 SDA
│ GPIO22 (SCL)────→ MPU6050 SCL
│              │
│ GPIO18 ─────────→ Motor Driver IN1
│ GPIO19 ─────────→ Motor Driver IN2
│              │
│ GPIO25 ─────────→ Green LED (with 220Ω resistor)
│ GPIO26 ─────────→ Red LED (with 220Ω resistor)
│              │
└──────────────┘

MOTOR DRIVER (L293D):
Pin 1 (Enable) → 5V
Pin 2 (IN1) → ESP32 GPIO18
Pin 3 (OUT1) → Motor Terminal 1
Pin 4 (GND) → Ground
Pin 5 (GND) → Ground
Pin 6 (OUT2) → Motor Terminal 2
Pin 7 (IN2) → ESP32 GPIO19
Pin 8 (VS) → 5V External Supply
Pin 16 (VCC) → 5V
```

### Gateway Node Connections

```
ESP32 GATEWAY:
┌──────────────┐
│   ESP32 #4   │
│              │
│ 3V3 ────────────→ OLED VCC
│ GND ────────────→ OLED GND
│ GPIO21 (SDA)────→ OLED SDA
│ GPIO22 (SCL)────→ OLED SCL
│              │
└──────────────┘
```

---

## 5. HOW IT WORKS

### Data Flow

1. **Sensing**: MPU6050 measures motor vibration (X, Y, Z axes)
2. **Processing**: ESP32 calculates vibration magnitude and compares to threshold
3. **Detection**: If magnitude exceeds threshold, anomaly is flagged
4. **Communication**: Node sends status packet via ESP-NOW to gateway
5. **Aggregation**: Gateway receives and processes all node statuses
6. **Explanation**: Gateway analyzes anomaly pattern and generates explanation
7. **Display**: Results shown on OLED with actionable information

### Anomaly Detection Logic

```
Normal Operation: Vibration magnitude < 1.5g
Warning Level: Vibration magnitude 1.5g - 2.5g
Critical Anomaly: Vibration magnitude > 2.5g
```

### Explanation Generation

The gateway identifies anomaly types based on patterns:

-  **Continuous high vibration** → "Possible bearing wear"
-  **Periodic spikes** → "Imbalance detected"
-  **Gradual increase** → "Mounting looseness"

---

## 6. BUILD INSTRUCTIONS

### Step 1: Prepare Hardware

1. Mount motors on stable base (wood/cardboard) ~15cm apart
2. Attach MPU6050 sensors near each motor base
3. Create motor faults:
   -  Motor 1: Keep normal
   -  Motor 2: Glue small weight to one propeller blade
   -  Motor 3: Add tape to create drag

### Step 2: Wire Sensor Nodes

For each sensor node:

1. Place ESP32 on breadboard
2. Connect MPU6050 using I2C (4 wires)
3. Wire motor through L293D driver
4. Add status LEDs with resistors
5. Connect power (USB for ESP32, external 5V for motors)

### Step 3: Wire Gateway

1. Place ESP32 #4 on separate breadboard
2. Connect OLED display via I2C
3. Power via USB

### Step 4: Program Devices

1. Upload sensor node code to ESP32 #1, #2, #3
2. Upload gateway code to ESP32 #4
3. Note MAC addresses for ESP-NOW pairing

---

## 7. CODE STRUCTURE

### Sensor Node Code Components

```cpp
// Main sections needed:
1. ESP-NOW initialization
2. MPU6050 sensor setup
3. Motor control functions
4. Anomaly detection algorithm
5. Wireless transmission
6. LED status control

// Key Libraries:
#include <WiFi.h>
#include <esp_now.h>
#include <Wire.h>
#include <MPU6050.h>

// Data Structure:
typedef struct {
    int nodeID;
    bool isAnomalous;
    float vibrationLevel;
    unsigned long timestamp;
} SensorMessage;
```

### Gateway Code Components

```cpp
// Main sections needed:
1. ESP-NOW receiver setup
2. OLED display initialization
3. Message reception handler
4. Status aggregation
5. Explanation generation
6. Display update functions

// Key Libraries:
#include <WiFi.h>
#include <esp_now.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>

// Node status tracking:
struct NodeStatus {
    bool isOnline;
    bool isAnomalous;
    float lastVibration;
    unsigned long lastSeen;
};
```

---

## 8. TESTING SEQUENCE

### Phase 1: Component Testing

1. Test each MPU6050 separately (read raw values)
2. Test motor control (forward/reverse)
3. Test LED indicators
4. Test OLED display (show text)

### Phase 2: Communication Testing

1. Get MAC address of gateway ESP32
2. Test ESP-NOW send from one node
3. Verify gateway receives messages
4. Test with all three nodes

### Phase 3: Integration Testing

1. Run all motors normally - verify green LEDs
2. Create vibration on motor 2 - verify anomaly detection
3. Check gateway display shows correct node and explanation
4. Test system resilience by disconnecting one node

---

## 9. TROUBLESHOOTING GUIDE

### Common Issues and Solutions

**Issue: MPU6050 not detected**

-  Check I2C connections (SDA to GPIO21, SCL to GPIO22)
-  Add 10kΩ pull-up resistors on SDA/SCL
-  Verify 3.3V power supply

**Issue: ESP-NOW messages not received**

-  Verify MAC address is correct (no typos)
-  Ensure all ESP32s are in WIFI_STA mode
-  Check distance between nodes (<10m recommended)

**Issue: Motor doesn't spin**

-  Check motor driver connections
-  Verify external 5V power supply
-  Test motor directly with battery

**Issue: Erratic sensor readings**

-  Keep motors and sensors physically separated
-  Add 100nF capacitor across motor terminals
-  Use shielded cables for I2C if possible

**Issue: OLED display blank**

-  Verify I2C address (usually 0x3C)
-  Check power connections
-  Try I2C scanner sketch first

---

## 10. DEMONSTRATION SCRIPT

### Setup (1 minute)

"This system monitors industrial equipment health using distributed edge AI..."

### Normal Operation (30 seconds)

-  All motors running
-  All LEDs green
-  Gateway shows "System Healthy"

### Fault Introduction (1 minute)

-  Motor 2 develops imbalance
-  Node 2 LED turns red
-  Gateway displays:
   ```
   ALERT: Node 2
   Status: Anomaly
   Type: Imbalance
   Action: Check mount
   ```

### Network Resilience (30 seconds)

-  Disconnect Node 1
-  System continues monitoring Nodes 2 & 3
-  Gateway shows "Node 1 Offline"

### Conclusion (30 seconds)

"Early detection prevents failures, saving costs and downtime..."

---

## 11. SAFETY NOTES

-  Never connect motor directly to ESP32 pins (use driver)
-  Ensure common ground between all power supplies
-  Motors may get warm during extended operation
-  Keep loose wires away from spinning propellers
-  Use appropriate current-rated wires for motor power

---

## 12. QUICK REFERENCE

### Pin Assignments

-  I2C: GPIO21 (SDA), GPIO22 (SCL)
-  Motor Control: GPIO18, GPIO19
-  Status LEDs: GPIO25 (Green), GPIO26 (Red)

### Power Requirements

-  ESP32: 5V via USB (draws ~200mA)
-  Motors: 5V external (each draws ~300mA)
-  Sensors: 3.3V from ESP32 (minimal current)

### Critical Parameters

-  Sampling Rate: 100Hz
-  Vibration Threshold: 1.5g
-  ESP-NOW Channel: 1
-  Communication Interval: 1 second

This document provides everything needed to build and demonstrate the system successfully. Follow the steps sequentially and test each component before integration.
