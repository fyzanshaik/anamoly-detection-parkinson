#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <esp_log.h>

#define SDA_PIN 21
#define SCL_PIN 22

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

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("Connecting to MPU6050...");
  for (int i = 0; i < 10; i++) {
    if (tryConnectMPU()) { mpuOk = true; break; }
    Serial.print(".");
    delay(300);
  }
  Serial.println(mpuOk ? "OK" : "FAILED");
}

void loop() {
  if (!mpuOk) {
    if (tryConnectMPU()) {
      mpuOk = true;
      Serial.println("[MPU] Reconnected");
    } else {
      delay(500);
      return;
    }
  }

  sensors_event_t a, g, temp;
  if (!mpu.getEvent(&a, &g, &temp)) {
    mpuOk = false;
    return;
  }

  float mag = sqrt(a.acceleration.x * a.acceleration.x +
                   a.acceleration.y * a.acceleration.y +
                   a.acceleration.z * a.acceleration.z);

  Serial.printf("ax:%.2f ay:%.2f az:%.2f mag:%.2f\n",
    a.acceleration.x, a.acceleration.y, a.acceleration.z, mag);

  delay(200);
}
