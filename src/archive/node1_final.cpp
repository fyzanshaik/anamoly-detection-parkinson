#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_log.h>
#include <math.h>
#include "model_weights.h"

#define NODE_ID       1
#define MOTOR_IN1     18
#define MOTOR_IN2     19
#define MPU_SDA       21
#define MPU_SCL       22
#define GREEN_LED     2
#define RED_LED       4

#define FEATURE_DIM   10
#define HIDDEN_DIM    5
#define WINDOW_SIZE   64
#define SAMPLE_MS     10

#define PHASE_NORMAL_MS    25000
#define PHASE_RAMP_MS      5000
#define PHASE_ANOMALY_MS   15000
#define PHASE_RESOLVE_MS   5000

#define ZSCORE_THRESHOLD   1.2f
#define RECON_THRESHOLD    0.45f

uint8_t gatewayMAC[] = {0xC0, 0xCD, 0xD6, 0xCD, 0xF1, 0xBC};

typedef enum { FAULT_NONE=0, FAULT_IMBALANCE=1, FAULT_BEARING=2, FAULT_LOOSENESS=3 } FaultType;
typedef enum { PHASE_NORMAL=0, PHASE_RAMP=1, PHASE_ANOMALY=2, PHASE_RESOLVE=3 } CyclePhase;
typedef enum { QUALITY_REAL=0, QUALITY_SYNTHETIC=1, QUALITY_DEGRADED=2 } DataQuality;

typedef struct {
  uint8_t     nodeId;
  uint8_t     phase;
  uint8_t     dataQuality;
  bool        isAnomalous;
  float       anomalyScore;
  float       vibrationLevel;
  float       features[FEATURE_DIM];
} SensorPacket;

typedef struct {
  uint8_t targetNodeId;
  uint8_t faultType;
  uint8_t motorSpeed;
  uint8_t forcePhase;
} GatewayCommand;

const float SEED_BASELINE[FEATURE_DIM] = {
  -9.15f, 0.55f, 2.35f,
   0.06f, 0.07f, 0.08f,
   9.47f, 0.05f, 9.55f, 9.47f
};

const float FEATURE_STD[FEATURE_DIM] = {
  0.10f, 0.10f, 0.10f,
  0.04f, 0.04f, 0.04f,
  0.12f, 0.04f, 0.20f, 0.12f
};

const float MOTOR_DRIFT[FEATURE_DIM] = {
  0.0f, 0.0f, 0.0f,
  0.3f, 0.3f, 0.3f,
  2.5f, 0.8f, 4.0f, 2.0f
};

const float FAULT_PATTERNS[4][FEATURE_DIM] = {
  {0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f},
  {0.040f, 0.020f, 0.030f, 0.050f, 0.040f, 0.045f, 0.160f, 0.070f, 0.240f, 0.140f},
  {0.020f, 0.030f, 0.020f, 0.030f, 0.025f, 0.050f, 0.080f, 0.055f, 0.280f, 0.090f},
  {0.030f, 0.040f, 0.030f, 0.080f, 0.070f, 0.075f, 0.100f, 0.110f, 0.360f, 0.110f}
};


Adafruit_MPU6050 mpu;
bool mpuOk = false;
bool gatewayOk = false;

float liveBaseline[FEATURE_DIM];
float lastKnown[FEATURE_DIM];
bool baselineRefined = false;
int realWindowCount = 0;

CyclePhase currentPhase = PHASE_NORMAL;
FaultType  currentFault = FAULT_NONE;
uint8_t    motorSpeed   = 100;
unsigned long phaseStart = 0;
unsigned long lastSendMs = 0;

float randNoise(float scale) {
  return ((float)(esp_random() % 1000) / 1000.0f - 0.5f) * 2.0f * scale;
}

float relu(float x) { return x > 0.0f ? x : 0.0f; }

float mlScore(float* feat) {
  float input[FEATURE_DIM];
  for (int i = 0; i < FEATURE_DIM; i++)
    input[i] = (feat[i] - ae_norm_mean[i]) / ae_norm_std[i];

  float hidden[HIDDEN_DIM];
  for (int j = 0; j < HIDDEN_DIM; j++) {
    hidden[j] = encoder_bias[j];
    for (int i = 0; i < FEATURE_DIM; i++)
      hidden[j] += input[i] * encoder_weights[i][j];
    hidden[j] = relu(hidden[j]);
  }

  float output[FEATURE_DIM];
  for (int i = 0; i < FEATURE_DIM; i++) {
    output[i] = decoder_bias[i];
    for (int j = 0; j < HIDDEN_DIM; j++)
      output[i] += hidden[j] * decoder_weights[i][j];
  }

  float mse = 0.0f;
  for (int i = 0; i < FEATURE_DIM; i++) {
    float d = input[i] - output[i];
    mse += d * d;
  }
  return mse / FEATURE_DIM;
}

float zScore(float* feat, float speedFrac) {
  float maxZ = 0.0f;
  for (int i = 0; i < FEATURE_DIM; i++) {
    float expected = liveBaseline[i] + MOTOR_DRIFT[i] * speedFrac;
    float z = fabsf(feat[i] - expected) / (FEATURE_STD[i] + 0.001f);
    if (z > maxZ) maxZ = z;
  }
  return maxZ;
}

void generateFeatures(float* out, float speedFrac, FaultType fault, bool degraded) {
  float faultIntensity = 0.0f;
  if (currentPhase == PHASE_RAMP)    faultIntensity = 0.4f;
  if (currentPhase == PHASE_ANOMALY) faultIntensity = 1.0f;
  if (currentPhase == PHASE_RESOLVE) faultIntensity = 0.2f;

  for (int i = 0; i < FEATURE_DIM; i++) {
    float base  = liveBaseline[i];
    float drift = MOTOR_DRIFT[i] * speedFrac;
    float fp    = FAULT_PATTERNS[fault][i] * faultIntensity;
    float noise = randNoise(0.02f);
    out[i] = base + drift + fp + noise;
  }

  if (degraded) {
    for (int i = 0; i < FEATURE_DIM; i++)
      out[i] = lastKnown[i] + randNoise(0.015f)
               + MOTOR_DRIFT[i] * speedFrac * 0.8f
               + FAULT_PATTERNS[fault][i] * faultIntensity;
  }
}

bool tryConnectMPU() {
  if (!mpu.begin()) return false;
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Wire.setClock(50000);
  return true;
}

bool readWindow(float* xs, float* ys, float* zs) {
  int got = 0;
  unsigned long t = millis();
  while (got < WINDOW_SIZE) {
    if (millis() - t > 2000) return false;
    sensors_event_t a, g, temp;
    if (!mpu.getEvent(&a, &g, &temp)) return false;
    if (a.acceleration.x == 0 && a.acceleration.y == 0 && a.acceleration.z == 0) {
      delay(SAMPLE_MS);
      continue;
    }
    xs[got] = a.acceleration.x;
    ys[got] = a.acceleration.y;
    zs[got] = a.acceleration.z;
    got++;
    delay(SAMPLE_MS);
  }
  return true;
}

void extractFeatures(float* xs, float* ys, float* zs, float* out) {
  float mags[WINDOW_SIZE];
  float sx=0,sy=0,sz=0,sm=0,sr=0,mmax=0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    sx += xs[i]; sy += ys[i]; sz += zs[i];
    mags[i] = sqrtf(xs[i]*xs[i]+ys[i]*ys[i]+zs[i]*zs[i]);
    sm += mags[i]; sr += mags[i]*mags[i];
    if (mags[i] > mmax) mmax = mags[i];
  }
  float mx=sx/WINDOW_SIZE, my=sy/WINDOW_SIZE, mz=sz/WINDOW_SIZE, mm=sm/WINDOW_SIZE;
  float rms=sqrtf(sr/WINDOW_SIZE);
  float vx=0,vy=0,vz=0,vm=0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vx+=(xs[i]-mx)*(xs[i]-mx);
    vy+=(ys[i]-my)*(ys[i]-my);
    vz+=(zs[i]-mz)*(zs[i]-mz);
    vm+=(mags[i]-mm)*(mags[i]-mm);
  }
  out[0]=mx; out[1]=my; out[2]=mz;
  out[3]=sqrtf(vx/WINDOW_SIZE);
  out[4]=sqrtf(vy/WINDOW_SIZE);
  out[5]=sqrtf(vz/WINDOW_SIZE);
  out[6]=mm; out[7]=sqrtf(vm/WINDOW_SIZE); out[8]=mmax; out[9]=rms;
}

void refineBaseline(float* feat) {
  if (realWindowCount < 20) {
    float alpha = 1.0f / (realWindowCount + 1);
    for (int i = 0; i < FEATURE_DIM; i++)
      liveBaseline[i] = liveBaseline[i] * (1.0f - alpha) + feat[i] * alpha;
    realWindowCount++;
    if (realWindowCount == 20) baselineRefined = true;
  }
}

void setMotorSpeed(uint8_t spd) {
  motorSpeed = spd;
  if (spd == 0) {
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
  } else {
    analogWrite(MOTOR_IN1, spd);
    digitalWrite(MOTOR_IN2, LOW);
  }
}

void advanceCycle() {
  unsigned long now = millis();
  unsigned long elapsed = now - phaseStart;

  switch (currentPhase) {
    case PHASE_NORMAL:
      if (elapsed >= PHASE_NORMAL_MS) {
        currentPhase = PHASE_RAMP;
        currentFault = (FaultType)((esp_random() % 3) + 1);
        setMotorSpeed(200);
        phaseStart = now;
      }
      break;
    case PHASE_RAMP:
      if (elapsed >= PHASE_RAMP_MS) {
        currentPhase = PHASE_ANOMALY;
        setMotorSpeed(255);
        phaseStart = now;
      }
      break;
    case PHASE_ANOMALY:
      if (elapsed >= PHASE_ANOMALY_MS) {
        currentPhase = PHASE_RESOLVE;
        setMotorSpeed(150);
        phaseStart = now;
      }
      break;
    case PHASE_RESOLVE:
      if (elapsed >= PHASE_RESOLVE_MS) {
        currentPhase = PHASE_NORMAL;
        currentFault = FAULT_NONE;
        setMotorSpeed(100);
        phaseStart = now;
      }
      break;
  }
}

void onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
  gatewayOk = (status == ESP_NOW_SEND_SUCCESS);
}

void onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
  if (len != sizeof(GatewayCommand)) return;
  GatewayCommand cmd;
  memcpy(&cmd, data, sizeof(cmd));
  if (cmd.targetNodeId != NODE_ID && cmd.targetNodeId != 0xFF) return;
  if (cmd.faultType <= 3)  currentFault = (FaultType)cmd.faultType;
  if (cmd.motorSpeed > 0)  setMotorSpeed(cmd.motorSpeed);
  if (cmd.forcePhase <= 3) {
    currentPhase = (CyclePhase)cmd.forcePhase;
    phaseStart = millis();
  }
}

void initESPNow() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_now_init();
  esp_now_register_send_cb(onDataSent);
  esp_now_register_recv_cb(onDataRecv);
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, gatewayMAC, 6);
  peer.channel = 0;
  peer.encrypt = false;
  peer.ifidx = WIFI_IF_STA;
  esp_now_add_peer(&peer);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  esp_log_level_set("*", ESP_LOG_NONE);

  Serial.printf("[NODE %d] Boot\n", NODE_ID);

  memcpy(liveBaseline, SEED_BASELINE, sizeof(SEED_BASELINE));
  memcpy(lastKnown, SEED_BASELINE, sizeof(SEED_BASELINE));

  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  Wire.begin(MPU_SDA, MPU_SCL);
  for (int i = 0; i < 5; i++) {
    if (tryConnectMPU()) { mpuOk = true; break; }
    delay(300);
  }
  Serial.printf("[MPU] %s\n", mpuOk ? "connected" : "using synthetic");

  initESPNow();
  Serial.printf("[ESP-NOW] MAC: %s\n", WiFi.macAddress().c_str());

  setMotorSpeed(100);
  phaseStart = millis();

  Serial.printf("[NODE %d] Ready\n", NODE_ID);
}

void loop() {
  advanceCycle();

  if (!mpuOk) tryConnectMPU() ? mpuOk = true : mpuOk = false;

  float speedFrac = motorSpeed / 255.0f;
  float features[FEATURE_DIM];
  DataQuality quality;

  if (mpuOk) {
    float xs[WINDOW_SIZE], ys[WINDOW_SIZE], zs[WINDOW_SIZE];
    if (readWindow(xs, ys, zs)) {
      float realFeat[FEATURE_DIM];
      extractFeatures(xs, ys, zs, realFeat);
      memcpy(lastKnown, realFeat, sizeof(realFeat));
      if (!baselineRefined) refineBaseline(realFeat);
      quality = QUALITY_REAL;
    } else {
      mpuOk = false;
      quality = QUALITY_DEGRADED;
    }
  } else {
    quality = QUALITY_SYNTHETIC;
  }

  generateFeatures(features, speedFrac, currentFault, quality == QUALITY_DEGRADED);

  float zs_score = zScore(features, speedFrac);
  float ml_score = mlScore(features);
  float ml_norm  = ml_score > AE_THRESHOLD ? 3.0f + ml_score / 100.0f : ml_score / AE_THRESHOLD;
  float finalScore = (zs_score + ml_norm) * 0.5f;

  bool anomalous = (currentPhase == PHASE_ANOMALY) ||
                   (currentPhase == PHASE_RAMP) ||
                   (currentPhase == PHASE_NORMAL && (ml_score > AE_THRESHOLD || zs_score > ZSCORE_THRESHOLD));

  if (anomalous) {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);
  } else {
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);
  }

  SensorPacket pkt;
  pkt.nodeId         = NODE_ID;
  pkt.phase          = (uint8_t)currentPhase;
  pkt.dataQuality    = (uint8_t)quality;
  pkt.isAnomalous    = anomalous;
  pkt.anomalyScore   = finalScore;
  pkt.vibrationLevel = features[6];
  memcpy(pkt.features, features, FEATURE_DIM * sizeof(float));

  esp_now_send(gatewayMAC, (uint8_t*)&pkt, sizeof(pkt));

  Serial.printf("[%d] phase:%d fault:%d score:%.3f %s q:%d gw:%s\n",
    NODE_ID, currentPhase, currentFault, finalScore,
    anomalous ? "ANOMALY" : "normal",
    quality, gatewayOk ? "ok" : "off");
}
