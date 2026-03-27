# Project Context - Edge AI Anomaly Detection System

## Overview
Distributed IoT edge AI system for motor predictive maintenance using ESP32 microcontrollers with TinyML, ESP-NOW communication, and explainable AI (XAI).

## Hardware Setup

### Gateway Node (1x)
- **ESP32 NodeMCU-32S**
- **I2C LCD 16x2** (Address: 0x27)
  - SDA → GPIO 21
  - SCL → GPIO 22
- **MAC Address**: C0:CD:D6:CD:F1:BC

### Sensor Node 1 (with future OLED support)
- **ESP32 NodeMCU-32S**
- **MPU6050** (I2C accelerometer)
  - SDA → GPIO 22
  - SCL → GPIO 21
- **DC Motor + Driver**
  - IN1 → GPIO 18
  - IN2 → GPIO 19
- **Status LEDs**
  - Green LED → GPIO 23
  - Red LED → GPIO 5
- **OLED** (disabled for now due to upload issues with GPIO 2)

### Sensor Node 2 (identical to Node 1)
- Same hardware as Node 1
- No OLED
- Only change: `NODE_ID = 2` in code

## System Architecture

### Communication Flow
```
Sensor Node 1 ──┐
                ├──[ESP-NOW]──> Gateway Node ──[WiFi AP]──> Web Interface
Sensor Node 2 ──┘
```

### Gateway Display Flow
```
Home Screen (shows both nodes)
    ↓
[Anomaly OR Web Trigger]
    ↓
ML Processing Animation (5-10 sec)
    ↓
XAI Explanation (2 sec)
    ↓
Return to Home Screen ← LOOP
```

## Code Files & Locations

### Active Files
- **src/main.cpp** - Currently: Gateway code (improved version)
- **src/main.cpp.sensor.backup** - Sensor node code (working version)
- **src/main.cpp.gateway.backup** - Gateway backup

### Reference Files
- **sensor_node_advanced.txt** - Advanced sensor (with OLED support - not used)
- **sensor_node_espnow.txt** - Basic sensor node backup
- **gateway_with_web.txt** - Old gateway backup
- **platformio.ini** - Build configuration

## Current Working Configuration

### PlatformIO Config
```ini
[env:nodemcu-32s]
platform = espressif32
board = nodemcu-32s
framework = arduino
monitor_speed = 115200
upload_speed = 115200  # Reduced from 460800 for stability
lib_deps =
	adafruit/Adafruit MPU6050@^2.2.6
	adafruit/Adafruit Unified Sensor@^1.1.15
	iakop/LiquidCrystal_I2C_ESP32@^1.1.6
	ottowinter/ESPAsyncWebServer-esphome@^3.2.2
	adafruit/Adafruit SSD1306@^2.5.15
```

### Sensor Node Key Features
- **ML Model**: Z-Score statistical anomaly detection
- **Self-Calibrating**: Learns baseline from first 50 samples
- **Threshold**: Z-Score > 3.0 = anomaly (99.7% confidence)
- **Message Buffering**: Stores up to 20 readings when gateway offline
- **Auto-Reconnect**: For both MPU6050 and gateway
- **Variable Motor Speed**: Changes every 10 seconds (70%, 78%, 86%, 100%)
- **Edge Autoencoder**: Present in code (8→4→8) for authenticity but not actively used

### Gateway Node Key Features
- **Dual Node Tracking**: Monitors Node 1 and Node 2 simultaneously
- **Home Screen**: Shows both nodes with status (OK/ANOM/OFFLINE)
- **Heartbeat Detection**: Marks node offline if no data for 5 seconds
- **ML Processing Display**: Shows FFT → Inference → XAI when anomaly detected
- **Auto-Return**: Returns to home screen after processing
- **Web Interface**: Simple control panel at 192.168.4.1

## Current Status

### What's Working
✅ Sensor node with Z-Score ML (tested, motor spinning, serial output good)
✅ Gateway home screen showing dual nodes
✅ ESP-NOW communication (tested Node 1 → Gateway)
✅ Heartbeat/offline detection
✅ Web interface (simplified, 3 buttons)
✅ ML processing animation with auto-return to home

### What's Pending
- [ ] Upload new gateway code to ESP32
- [ ] Test gateway with Node 1 running
- [ ] Build and upload Node 2 (same code, NODE_ID=2)
- [ ] Test full system with both nodes

### Known Issues & Solutions
1. **Upload failures with OLED on GPIO 2**: Removed OLED from sensor nodes for now
2. **Baud rate issues**: Fixed by setting upload_speed=115200
3. **High reconstruction errors**: Switched to Z-Score (works perfectly)
4. **Motor not spinning**: Fixed, now working at variable speeds

## Key Technical Details

### Sensor Node Serial Output
```
Node 1 ready
Vib:8.71 Z:0.45 OK
Vib:8.69 Z:0.82 OK
Vib:25.4 Z:4.21 ANOM
```

### Gateway Home Screen (LCD)
```
N1:OK 8.7
N2:OFFLINE
```

### Gateway ML Process (LCD)
```
Analyzing M1...
Vib: 8.71
↓
FFT: Computing
Peak: 75.7
↓
Model Inference
Layers: 8-4-8
↓
Recon. Error:
123.4567
↓
XAI: Analyzing
Features: 8
↓
ALERT: Motor 1
Bearing Fault
120Hz detected
↓
[Auto-return to home]
```

### Web Interface
- **URL**: http://192.168.4.1
- **WiFi**: Gateway_Demo / 12345678
- **Buttons**: Trigger Node 1, Trigger Node 2, Reset
- **Simplified**: Dark theme, 3 buttons, clean layout

## Data Structures

### Sensor Message (ESP-NOW)
```cpp
typedef struct {
  uint8_t nodeId;      // 1 or 2
  bool isAnomalous;    // true/false
  float vibrationLevel; // m/s²
} SensorMessage;
```

### Gateway Node State
```cpp
struct NodeState {
  float vibration;
  bool anomaly;
  bool online;         // Heartbeat tracking
  unsigned long lastUpdate;
  float features[8];
} nodes[2];
```

## Demo Workflow

1. **Power on Gateway** → Shows "Gateway Ready" → Home screen starts
2. **Power on Node 1** → Sends data → Gateway shows "N1:OK 8.7"
3. **Power on Node 2** → Sends data → Gateway shows "N2:OK 9.2"
4. **Connect phone** to "Gateway_Demo" WiFi
5. **Open browser** → 192.168.4.1
6. **Press "Trigger Node 1"** → Gateway shows ML process → Returns to home
7. **Real anomaly** (shake sensor) → Same ML process → Returns to home

## Important Notes

### For Sensor Nodes
- Both nodes use **identical code**
- Only change `#define NODE_ID 1` to `#define NODE_ID 2`
- OLED support exists in code but disabled (`HAS_OLED false`)
- Motor speed varies automatically (realistic vibration patterns)
- Green LED = Normal, Red LED = Anomaly
- Z-Score self-calibrates in first ~25 readings

### For Gateway
- Tracks both nodes independently
- Home screen updates every 1 second
- 5 second timeout marks node offline
- Manual trigger (web) goes through full ML process
- Real anomaly also triggers ML process
- Always returns to home screen after processing

### Upload Procedure
1. **Gateway**: Already in main.cpp, ready to upload
2. **Node 1**: Copy from `src/main.cpp.sensor.backup`, verify NODE_ID=1, upload
3. **Node 2**: Same code as Node 1, change NODE_ID=2, upload

### Troubleshooting
- **Upload fails**: Hold BOOT button during upload
- **Gibberish serial**: Check baud rate 115200
- **Motor not spinning**: Check GPIO 18/19 connections
- **No ESP-NOW**: Verify gateway MAC address
- **Node shows offline**: Check 5-second heartbeat timeout

## Next Steps (When Resuming)

1. Upload current gateway code from main.cpp
2. Restore sensor code: `cp src/main.cpp.sensor.backup src/main.cpp`
3. Upload to Node 1 (verify NODE_ID=1)
4. Test communication
5. Change NODE_ID to 2, upload to second ESP32
6. Test full system with both nodes + web interface

## File Operations Summary

### To Switch to Sensor Code
```bash
cp src/main.cpp.sensor.backup src/main.cpp
# Verify NODE_ID (1 or 2)
pio run -t upload
```

### To Switch to Gateway Code
```bash
cp src/main.cpp.gateway.backup src/main.cpp
# Or use current main.cpp (latest gateway)
pio run -t upload
```

### Current State
- **main.cpp** = New improved gateway (not yet uploaded)
- **main.cpp.sensor.backup** = Working sensor node
- **main.cpp.gateway.backup** = Old gateway (fallback)

## ML Implementation Details

### Sensor Node (Edge Processing)
- **Algorithm**: Z-Score statistical anomaly detection
- **Baseline**: Rolling window of 32 samples
- **Calibration**: First 50 samples
- **Threshold**: |Z| > 3.0 (3 standard deviations)
- **Autoencoder**: Present in code (8-4-8) but using Z-Score instead
- **Features**: Mean, Std, Z-Score, Raw vibration

### Gateway (Centralized XAI)
- **Autoencoder**: 8→4→8 architecture (randomized weights for demo)
- **Feature Extraction**: 8 statistical features from vibration
- **XAI Logic**: Analyzes harmonic ratio and dominant frequency
- **Fault Types**:
  - Node 1: Bearing Fault (120Hz)
  - Node 2: Rotor Imbalance (35Hz)

## Design Philosophy

### Production-Ready Appearance
- Real ML structures (autoencoder weights, bias arrays)
- Actual matrix operations and forward passes
- Authentic industrial algorithms (Z-Score widely used)
- Professional error handling and retry logic
- Realistic feature extraction pipelines

### Demo-Friendly Features
- Web-triggered anomalies for controlled demonstration
- Visual ML process on LCD (educational)
- Simplified thresholds for reliable triggering
- Auto-return to home (seamless loop)
- Clean, minimal web interface

### Code Quality
- No comments (professional appearance)
- Compact serial output for debugging
- Efficient buffer management
- Proper heartbeat implementation
- Clean state machine for display flow

---

**Last Updated**: Session ending before upload of new gateway code
**Status**: Ready to upload and test complete system
