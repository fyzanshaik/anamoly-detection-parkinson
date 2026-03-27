#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define MOTOR_IN1 18
#define MOTOR_IN2 19
#define MPU_SDA   22
#define MPU_SCL   21

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);

  Wire.begin(MPU_SDA, MPU_SCL);
  Wire.setClock(50000); // slow I2C to reduce noise sensitivity
  if (!mpu.begin()) {
    Serial.println("MPU6050 NOT FOUND - check wiring!");
    while (1) delay(500);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("MPU6050 found. Starting...");
  Serial.println("speed,ax,ay,az,magnitude");
}

void printReading(const char* label) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float mag = sqrt(a.acceleration.x * a.acceleration.x +
                   a.acceleration.y * a.acceleration.y +
                   a.acceleration.z * a.acceleration.z);
  Serial.printf("%s,%.3f,%.3f,%.3f,%.3f\n",
    label,
    a.acceleration.x,
    a.acceleration.y,
    a.acceleration.z,
    mag);
}

void loop() {
  // Motor OFF - baseline
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  for (int i = 0; i < 10; i++) {
    printReading("OFF");
    delay(100);
  }

  // Low speed
  analogWrite(MOTOR_IN1, 100);
  digitalWrite(MOTOR_IN2, LOW);
  delay(500); // settle
  for (int i = 0; i < 10; i++) {
    printReading("LOW");
    delay(100);
  }

  // High speed
  analogWrite(MOTOR_IN1, 255);
  digitalWrite(MOTOR_IN2, LOW);
  delay(500); // settle
  for (int i = 0; i < 10; i++) {
    printReading("HIGH");
    delay(100);
  }
}
