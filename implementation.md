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
