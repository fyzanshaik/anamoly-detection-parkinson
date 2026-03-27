// #include <Arduino.h>
// #include <Adafruit_MPU6050.h>
// #include <Adafruit_Sensor.h>
// #include <Wire.h>

// Adafruit_MPU6050 mpu;

// // State variable to track the connection status
// bool connection_lost = false;

// void setup() {
//   Serial.begin(115200);
//   delay(2000);
  
//   Serial.println("=== MPU6050 Test ===");
//   Serial.println("Initializing...");
  
//   Wire.begin(22, 21);
  
//   if (!mpu.begin()) {
//     Serial.println("ERROR: Failed to find MPU6050 chip!");
//     while (1) {
//       delay(1000);
//     }
//   }
  
//   Serial.println("SUCCESS: MPU6050 Found!");
//   Serial.println("");
  
//   mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
//   mpu.setGyroRange(MPU6050_RANGE_500_DEG);
//   mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
//   Serial.println("Reading sensor data...");
//   delay(1000);
// }

// void loop() {
//   sensors_event_t a, g, temp;
//   mpu.getEvent(&a, &g, &temp);

//   bool is_all_zero = (a.acceleration.x == 0.00 && a.acceleration.y == 0.00 && a.acceleration.z == 0.00 &&
//                       g.gyro.x == 0.00 && g.gyro.y == 0.00 && g.gyro.z == 0.00);

//   if (is_all_zero) {
//     // Only print the error and try to reconnect if this is the FIRST time we've detected the failure.
//     if (!connection_lost) {
//       connection_lost = true; // Set the flag to true to prevent this block from running again
//       Serial.println("Accel: 0.00, 0.00, 0.00 | Gyro: 0.00, 0.00, 0.00");
//       Serial.println("Communication lost. Will attempt to reconnect silently...");
//     }

//     // Keep trying to reconnect in the background on each loop
//     if (mpu.begin()) {
//       Serial.println("SUCCESS: Reconnected to MPU6050!");
//       mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
//       mpu.setGyroRange(MPU6050_RANGE_500_DEG);
//       mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
//       connection_lost = false; // Reset the flag on successful reconnection
//     }
//     delay(500); // Wait before the next reconnection attempt
//     return;
//   }
  
//   // If we are here, it means the connection is fine
//   connection_lost = false; // Ensure flag is reset if connection recovers on its own

//   // Normal data printing
//   Serial.print("Accel: ");
//   Serial.print(a.acceleration.x, 2);
//   Serial.print(", ");
//   Serial.print(a.acceleration.y, 2);
//   Serial.print(", ");
//   Serial.print(a.acceleration.z, 2);
//   Serial.print(" | ");
  
//   Serial.print("Gyro: ");
//   Serial.print(g.gyro.x, 2);
//   Serial.print(", ");
//   Serial.print(g.gyro.y, 2);
//   Serial.print(", ");
//   Serial.println(g.gyro.z, 2);
  
//   delay(500);
// }