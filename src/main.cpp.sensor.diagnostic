#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define NODE_ID 1

#define MOTOR_IN1 18
#define MOTOR_IN2 19
#define GREEN_LED 23
#define RED_LED 5

#define MPU_SDA 22
#define MPU_SCL 21

#define VIBRATION_THRESHOLD 20.0
#define WINDOW_SIZE 32
#define BUFFER_SIZE 20
#define BASELINE_SAMPLES 50
#define INPUT_DIM 8
#define HIDDEN_DIM 4

uint8_t gatewayMAC[] = {0xC0, 0xCD, 0xD6, 0xCD, 0xF1, 0xBC};

typedef struct {
  uint8_t nodeId;
  bool isAnomalous;
  float vibrationLevel;
} SensorMessage;

struct MessageBuffer {
  SensorMessage messages[BUFFER_SIZE];
  int writeIndex;
  int readIndex;
  int count;
} msgBuffer = {.writeIndex = 0, .readIndex = 0, .count = 0};

struct EdgeAutoencoder {
  float encoder_weights[INPUT_DIM][HIDDEN_DIM];
  float decoder_weights[HIDDEN_DIM][INPUT_DIM];
  float encoder_bias[HIDDEN_DIM];
  float decoder_bias[INPUT_DIM];
} edgeModel;

struct MLModel {
  float baseline_mean;
  float baseline_std;
  float vibrationWindow[WINDOW_SIZE];
  int windowIndex;
  int sampleCount;
  bool calibrated;
} mlModel;

struct SensorState {
  float features[INPUT_DIM];
  bool mpuConnected;
  bool gatewayConnected;
  unsigned long lastSuccessfulSend;
  int failedSendCount;
  int motorSpeed;
  float anomalyScore;
} sensorState;

Adafruit_MPU6050 mpu;

void initializeEdgeModel() {
  Serial.println("[INIT] Initializing ML model...");
  randomSeed(analogRead(0) + NODE_ID * 100);
  for(int i = 0; i < INPUT_DIM; i++) {
    for(int j = 0; j < HIDDEN_DIM; j++) {
      edgeModel.encoder_weights[i][j] = random(-100, 100) / 100.0;
    }
    edgeModel.decoder_bias[i] = 0.1;
  }
  for(int i = 0; i < HIDDEN_DIM; i++) {
    edgeModel.encoder_bias[i] = 0.1;
    for(int j = 0; j < INPUT_DIM; j++) {
      edgeModel.decoder_weights[i][j] = random(-100, 100) / 100.0;
    }
  }

  mlModel.windowIndex = 0;
  mlModel.sampleCount = 0;
  mlModel.calibrated = false;
  mlModel.baseline_mean = 0;
  mlModel.baseline_std = 0;
  Serial.println("[INIT] ML model initialized (8-4-8 architecture)");
}

void updateBaseline(float vibration) {
  mlModel.vibrationWindow[mlModel.windowIndex] = vibration;
  mlModel.windowIndex = (mlModel.windowIndex + 1) % WINDOW_SIZE;
  mlModel.sampleCount++;

  if(mlModel.sampleCount >= BASELINE_SAMPLES && !mlModel.calibrated) {
    float sum = 0;
    for(int i = 0; i < WINDOW_SIZE; i++) {
      sum += mlModel.vibrationWindow[i];
    }
    mlModel.baseline_mean = sum / WINDOW_SIZE;

    float variance = 0;
    for(int i = 0; i < WINDOW_SIZE; i++) {
      float diff = mlModel.vibrationWindow[i] - mlModel.baseline_mean;
      variance += diff * diff;
    }
    mlModel.baseline_std = sqrt(variance / WINDOW_SIZE);
    mlModel.calibrated = true;
    Serial.println("[ML] Baseline calibrated!");
  }
}

float calculateAnomalyScore(float vibration) {
  if(!mlModel.calibrated) {
    updateBaseline(vibration);
    return 0.0;
  }

  float z_score = abs((vibration - mlModel.baseline_mean) / (mlModel.baseline_std + 0.01));
  sensorState.features[0] = mlModel.baseline_mean;
  sensorState.features[1] = mlModel.baseline_std;
  sensorState.features[2] = z_score;
  sensorState.features[3] = vibration;

  return z_score;
}

void bufferMessage(SensorMessage msg) {
  if(msgBuffer.count < BUFFER_SIZE) {
    msgBuffer.messages[msgBuffer.writeIndex] = msg;
    msgBuffer.writeIndex = (msgBuffer.writeIndex + 1) % BUFFER_SIZE;
    msgBuffer.count++;
    Serial.print("[BUFFER] Message stored (");
    Serial.print(msgBuffer.count);
    Serial.println(" in queue)");
  } else {
    msgBuffer.readIndex = (msgBuffer.readIndex + 1) % BUFFER_SIZE;
    msgBuffer.messages[msgBuffer.writeIndex] = msg;
    msgBuffer.writeIndex = (msgBuffer.writeIndex + 1) % BUFFER_SIZE;
    Serial.println("[BUFFER] Buffer full, dropping oldest");
  }
}

bool sendBufferedMessages() {
  if(msgBuffer.count == 0) return true;
  int sent = 0;
  while(msgBuffer.count > 0 && sent < 3) {
    SensorMessage msg = msgBuffer.messages[msgBuffer.readIndex];
    esp_err_t result = esp_now_send(gatewayMAC, (uint8_t*)&msg, sizeof(msg));
    if(result == ESP_OK) {
      msgBuffer.readIndex = (msgBuffer.readIndex + 1) % BUFFER_SIZE;
      msgBuffer.count--;
      sent++;
      delay(50);
    } else {
      return false;
    }
  }
  Serial.print("[BUFFER] Sent ");
  Serial.print(sent);
  Serial.println(" buffered messages");
  return true;
}

void onDataSent(const uint8_t *mac, esp_now_send_status_t status) {
  if(status == ESP_NOW_SEND_SUCCESS) {
    sensorState.gatewayConnected = true;
    sensorState.lastSuccessfulSend = millis();
    sensorState.failedSendCount = 0;
    Serial.println("[ESP-NOW] Send SUCCESS");
    if(msgBuffer.count > 0) {
      sendBufferedMessages();
    }
  } else {
    sensorState.failedSendCount++;
    Serial.print("[ESP-NOW] Send FAILED (attempt ");
    Serial.print(sensorState.failedSendCount);
    Serial.println(")");
    if(sensorState.failedSendCount >= 3) {
      sensorState.gatewayConnected = false;
      Serial.println("[ESP-NOW] Gateway offline");
    }
  }
}

void adjustMotorSpeed() {
  static unsigned long lastSpeedChange = 0;
  if(millis() - lastSpeedChange > 10000) {
    int speeds[] = {180, 200, 220, 255};
    static int speedIndex = 0;
    speedIndex = (speedIndex + 1) % 4;
    sensorState.motorSpeed = (speeds[speedIndex] * 100) / 255;
    analogWrite(MOTOR_IN1, speeds[speedIndex]);
    digitalWrite(MOTOR_IN2, LOW);
    Serial.print("[MOTOR] Speed changed to ");
    Serial.print(sensorState.motorSpeed);
    Serial.println("%");
    lastSpeedChange = millis();
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n\n================================");
  Serial.println("  SENSOR NODE " + String(NODE_ID));
  Serial.println("  Diagnostic Version");
  Serial.println("================================\n");

  sensorState.mpuConnected = false;
  sensorState.gatewayConnected = false;
  sensorState.lastSuccessfulSend = 0;
  sensorState.failedSendCount = 0;
  sensorState.motorSpeed = 70;
  sensorState.anomalyScore = 0;

  initializeEdgeModel();

  Serial.println("[GPIO] Configuring pins...");
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);
  Serial.println("[GPIO] Pins configured");

  Serial.println("[WiFi] Setting up WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  Serial.print("[WiFi] MAC Address: ");
  Serial.println(WiFi.macAddress());

  Serial.println("[ESP-NOW] Initializing...");
  if(esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] INIT FAILED!");
    digitalWrite(RED_LED, HIGH);
    while(1) {
      Serial.println("[ERROR] ESP-NOW init failed, halted");
      delay(2000);
    }
  }
  Serial.println("[ESP-NOW] Initialized successfully");

  esp_now_register_send_cb(onDataSent);

  Serial.print("[ESP-NOW] Adding gateway peer: ");
  Serial.printf("%02X:%02X:%02X:%02X:%02X:%02X\n",
    gatewayMAC[0], gatewayMAC[1], gatewayMAC[2],
    gatewayMAC[3], gatewayMAC[4], gatewayMAC[5]);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, gatewayMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;

  if(esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("[ESP-NOW] ADD PEER FAILED!");
    digitalWrite(RED_LED, HIGH);
    while(1) {
      Serial.println("[ERROR] Failed to add gateway peer, halted");
      delay(2000);
    }
  }
  Serial.println("[ESP-NOW] Gateway peer added successfully");

  Serial.println("[MPU6050] Initializing I2C...");
  Wire.begin(MPU_SDA, MPU_SCL);
  delay(100);

  Serial.println("[MPU6050] Searching for sensor...");
  if(!mpu.begin()) {
    Serial.println("[MPU6050] NOT FOUND!");
    digitalWrite(RED_LED, HIGH);
    while(1) {
      Serial.println("[ERROR] MPU6050 not found, check wiring!");
      delay(2000);
    }
  }
  Serial.println("[MPU6050] Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  sensorState.mpuConnected = true;
  Serial.println("[MPU6050] Configured (Range: 8G, Gyro: 500deg/s, BW: 21Hz)");

  Serial.println("[MOTOR] Starting motor at 70% speed...");
  analogWrite(MOTOR_IN1, 180);
  digitalWrite(MOTOR_IN2, LOW);
  digitalWrite(GREEN_LED, HIGH);
  Serial.println("[MOTOR] Motor started");

  Serial.println("\n================================");
  Serial.println("  SYSTEM READY - MONITORING");
  Serial.println("================================\n");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  bool is_all_zero = (a.acceleration.x == 0.00 && a.acceleration.y == 0.00 &&
                      a.acceleration.z == 0.00 && g.gyro.x == 0.00 &&
                      g.gyro.y == 0.00 && g.gyro.z == 0.00);

  if(is_all_zero && sensorState.mpuConnected) {
    sensorState.mpuConnected = false;
    Serial.println("[MPU6050] Connection lost! Stopping motor...");
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);

    delay(500);
    Serial.println("[MPU6050] Attempting reconnection...");
    if(mpu.begin()) {
      Serial.println("[MPU6050] Reconnected!");
      mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
      mpu.setGyroRange(MPU6050_RANGE_500_DEG);
      mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
      sensorState.mpuConnected = true;
      analogWrite(MOTOR_IN1, 180);
      digitalWrite(MOTOR_IN2, LOW);
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(RED_LED, LOW);
    }
    return;
  }

  if(!sensorState.mpuConnected) {
    delay(500);
    return;
  }

  float vibration_magnitude = sqrt(
    a.acceleration.x * a.acceleration.x +
    a.acceleration.y * a.acceleration.y +
    a.acceleration.z * a.acceleration.z
  );

  float anomalyScore = calculateAnomalyScore(vibration_magnitude);
  sensorState.anomalyScore = anomalyScore;

  bool isAnomalous = (anomalyScore > 3.0) || (vibration_magnitude > VIBRATION_THRESHOLD);

  SensorMessage msg;
  msg.nodeId = NODE_ID;
  msg.isAnomalous = isAnomalous;
  msg.vibrationLevel = vibration_magnitude;

  esp_err_t result = esp_now_send(gatewayMAC, (uint8_t*)&msg, sizeof(msg));

  if(result != ESP_OK) {
    Serial.println("[ESP-NOW] Send error, buffering...");
    bufferMessage(msg);
  }

  Serial.print("[DATA] Vib:");
  Serial.print(vibration_magnitude, 2);
  Serial.print(" | Z:");
  Serial.print(anomalyScore, 2);
  Serial.print(" | ");
  Serial.print(isAnomalous ? "ANOM" : "OK");
  Serial.print(" | GW:");
  Serial.print(sensorState.gatewayConnected ? "CONN" : "DISC");
  Serial.print(" | Buf:");
  Serial.println(msgBuffer.count);

  if(isAnomalous) {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);
  } else {
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);
  }

  adjustMotorSpeed();

  if(sensorState.gatewayConnected &&
     (millis() - sensorState.lastSuccessfulSend > 15000)) {
    sensorState.gatewayConnected = false;
    Serial.println("[ESP-NOW] Gateway timeout");
  }

  delay(2000);
}
