#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <esp_log.h>

#define MPU_SDA 22
#define MPU_SCL 21

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  delay(500);
  esp_log_level_set("*", ESP_LOG_NONE);

  Serial.println("MPU6050 test starting...");
  Wire.begin(MPU_SDA, MPU_SCL);

  if (!mpu.begin()) {
    Serial.println("FAIL — MPU6050 not found. Check wiring.");
    Serial.println("Retrying every second...");
    while (!mpu.begin()) {
      Serial.print(".");
      delay(1000);
    }
  }

  Serial.println("OK — MPU6050 found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  Serial.printf("ax:%.2f ay:%.2f az:%.2f\n",
    a.acceleration.x, a.acceleration.y, a.acceleration.z);
  delay(200);
}
