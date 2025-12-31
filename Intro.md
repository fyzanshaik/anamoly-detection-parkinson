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
