#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <esp_log.h>

#define SDA_PIN   21
#define SCL_PIN   22
#define MOTOR_IN1 18
#define MOTOR_IN2 19

Adafruit_MPU6050 mpu;
bool mpuOk = false;

bool tryConnectMPU() {
  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  esp_log_level_set("*", ESP_LOG_NONE);

  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("Connecting MPU...");
  for (int i = 0; i < 10; i++) {
    if (tryConnectMPU()) { mpuOk = true; break; }
    delay(300);
  }
  Serial.println(mpuOk ? "MPU OK" : "MPU FAILED");
  Serial.println("speed,ax,ay,az,mag");
}

void loop() {
  // Silent reconnect
  if (!mpuOk) {
    if (tryConnectMPU()) mpuOk = true;
    else { delay(500); return; }
  }

  int speeds[] = {0, 150, 200, 255};
  const char* labels[] = {"OFF", "LOW", "HIGH", "FULL"};

  for (int s = 0; s < 4; s++) {
    if (speeds[s] == 0) {
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, LOW);
    } else {
      analogWrite(MOTOR_IN1, speeds[s]);
      digitalWrite(MOTOR_IN2, LOW);
    }
    delay(800); // settle

    // Take 10 readings at this speed
    for (int i = 0; i < 10; i++) {
      sensors_event_t a, g, temp;
      if (!mpu.getEvent(&a, &g, &temp)) { mpuOk = false; break; }
      float mag = sqrt(a.acceleration.x * a.acceleration.x +
                       a.acceleration.y * a.acceleration.y +
                       a.acceleration.z * a.acceleration.z);
      Serial.printf("%s,%.3f,%.3f,%.3f,%.3f\n",
        labels[s], a.acceleration.x, a.acceleration.y, a.acceleration.z, mag);
      delay(100);
    }
  }
}
