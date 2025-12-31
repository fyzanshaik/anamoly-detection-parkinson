# 🔧 Advanced Sensor Node - Setup Guide

## 📋 Hardware Configuration

### Node 1 (With OLED Display)
- **ESP32 NodeMCU-32S**
- **MPU6050** (Primary I2C):
  - SDA: GPIO 22
  - SCL: GPIO 21
- **OLED SSD1306** (Secondary I2C):
  - SDA: GPIO 2 (D4)
  - SCL: GPIO 5 (D5)
- **Motor Driver**:
  - IN1: GPIO 18
  - IN2: GPIO 19
- **Status LEDs**:
  - Green LED: GPIO 23
  - Red LED: GPIO 15 ⚠️ (Changed from GPIO 5 to avoid OLED conflict)
- **Gateway MAC**: `C0:CD:D6:CD:F1:BC`

### Node 2 (Without OLED)
- Same as Node 1 but:
  - Set `#define HAS_OLED false`
  - No OLED hardware needed
  - Red LED can use GPIO 15 (to match Node 1)

## 🚀 Key Features

### 1. **Edge ML Processing**
- **Autoencoder Model**: 8-4-8 architecture
- **Feature Extraction**: 8 statistical features
  - Mean, Peak, RMS
  - Skewness, Kurtosis
  - Dominant Frequency, Harmonic Ratio, Energy
- **Inference**: Runs on every sensor reading
- **Anomaly Detection**: Based on reconstruction error (threshold: 0.35)

### 2. **Robust ESP-NOW Communication**
- **Message Buffering**: Stores up to 20 readings when gateway offline
- **Automatic Retry**: Sends buffered messages when connection restored
- **Connection Monitoring**:
  - Tracks successful/failed sends
  - Detects gateway timeout (15 seconds)
  - Visual feedback on OLED and LEDs

### 3. **OLED Display (Node 1 Only)**
Shows real-time:
- Node ID & Gateway status
- Vibration level (m/s²)
- Reconstruction error
- Anomaly status
- Motor speed (%)
- Message buffer status

### 4. **MPU6050 Resilience**
- **Auto-reconnect**: Detects sensor disconnection
- **Motor Safety**: Stops motor when sensor fails
- **Visual Indicators**: Red LED during sensor failure
- **Serial Debugging**: Detailed connection status

### 5. **Variable Motor Speed**
- Automatically changes speed every 10 seconds
- Speeds: 70%, 78%, 86%, 100%
- Creates different vibration patterns for ML testing

### 6. **Heartbeat Mechanism**
- Pings gateway every 5 seconds
- Monitors connection health
- Serial output for debugging

## 📊 Serial Output Example

```
=================================
   SENSOR NODE 1
   Edge AI Vibration Monitor
=================================

[ML] Initializing edge autoencoder...
[ML] Model initialized: 8-4-8 architecture
[GPIO] Pins configured
[OLED] ✓ Display initialized
[WIFI] MAC Address: XX:XX:XX:XX:XX:XX
[ESP-NOW] ✓ Initialized
[ESP-NOW] ✓ Gateway peer added: C0:CD:D6:CD:F1:BC
[MPU6050] Searching for sensor...
[MPU6050] ✓ Found and configured
[MOTOR] ✓ Started at 70%

=================================
   SYSTEM READY - MONITORING
=================================

[DATA] Vib: 9.73 | Err: 0.1234 | Status: Normal | Gateway: Connected | Buffer: 0
[ESP-NOW] ✓ Send Success
[DATA] Vib: 11.45 | Err: 0.1567 | Status: Normal | Gateway: Connected | Buffer: 0
[ESP-NOW] ✓ Send Success
[MOTOR] Speed changed to 78%
[DATA] Vib: 14.22 | Err: 0.2890 | Status: Normal | Gateway: Connected | Buffer: 0
[ESP-NOW] ✓ Send Success
```

## 🔄 Error Handling Flow

### When Gateway Goes Offline:
1. **Detection**: 3 consecutive send failures
2. **Buffering**: Stores messages in circular buffer (20 max)
3. **Visual Feedback**:
   - OLED shows "GW:LOST"
   - Buffer count displayed
4. **Serial Output**: `[BUFFER] Stored message (X in queue)`

### When Gateway Reconnects:
1. **Auto-send**: Transmits buffered messages (max 3 per cycle)
2. **Progress Tracking**: Serial shows send progress
3. **Visual Update**: OLED shows "GW:OK"

### When MPU6050 Disconnects:
1. **Detection**: All-zero sensor readings
2. **Safety**: Motor stops immediately
3. **Visual**: Red LED on, Green LED off
4. **Recovery**: Auto-reconnect every 500ms
5. **Resume**: Motor restarts when sensor reconnected

## 🎯 Configuration for Node 2

To configure for the second node (without OLED):

```cpp
// Change these lines:
#define NODE_ID 2        // Change from 1 to 2
#define HAS_OLED false   // Change from true to false
```

That's it! The code automatically:
- Disables OLED initialization
- Skips display updates
- Uses different fault patterns (35Hz vs 120Hz)

## 📝 Key Code Locations

### Main Components:
- **Line 14**: `HAS_OLED` - Enable/disable OLED
- **Line 8**: `NODE_ID` - Set node identifier
- **Line 30-35**: ML model configuration
- **Line 106-132**: Edge autoencoder initialization
- **Line 134-165**: Feature extraction
- **Line 167-199**: ML inference engine
- **Line 201-230**: Message buffering system
- **Line 272-313**: OLED update function
- **Line 315-338**: ESP-NOW callback with retry logic
- **Line 340-355**: Motor speed variation

### Debug Points:
- **Line 344**: Gateway send status
- **Line 425**: MPU6050 connection check
- **Line 512**: Main data processing loop

## 🔧 Testing Checklist

- [ ] Upload code to Node 1 (with OLED)
- [ ] Verify OLED displays node info
- [ ] Check serial monitor for ML initialization
- [ ] Confirm MPU6050 detected
- [ ] Verify motor starts at 70%
- [ ] Test ESP-NOW send to gateway
- [ ] Simulate gateway offline (power off gateway)
- [ ] Verify buffering occurs
- [ ] Power on gateway
- [ ] Confirm buffered messages sent
- [ ] Upload to Node 2 with `HAS_OLED false`

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| OLED blank | Check I2C address (0x3C), verify SDA/SCL pins |
| Red LED conflict | Moved to GPIO 15, update wiring |
| ESP-NOW fails | Verify gateway MAC, check WiFi mode |
| MPU6050 not found | Check I2C wiring (GPIO 22/21) |
| Buffer overflow | Increase `BUFFER_SIZE` or reduce send interval |
| Motor not running | Check GPIO 18/19 connections |

## 📂 File Location

**Sensor Node Code**: `sensor_node_advanced.txt`

To use this code:
1. Copy content to `src/main.cpp`
2. Configure `NODE_ID` and `HAS_OLED`
3. Update `gatewayMAC` if needed
4. Upload to ESP32
5. Monitor serial output
