// #include <Arduino.h>

// #define MOTOR_IN1 18
// #define MOTOR_IN2 19
// #define GREEN_LED 23
// #define RED_LED 5

// void setup() {
//   Serial.begin(115200);
  
//   pinMode(MOTOR_IN1, OUTPUT);
//   pinMode(MOTOR_IN2, OUTPUT);
//   pinMode(GREEN_LED, OUTPUT);
//   pinMode(RED_LED, OUTPUT);
  
//   digitalWrite(MOTOR_IN1, LOW);
//   digitalWrite(MOTOR_IN2, LOW);
//   digitalWrite(GREEN_LED, LOW);
//   digitalWrite(RED_LED, LOW);
// }

// void loop() {
//   Serial.println("Motor FORWARD - Green LED");
//   digitalWrite(GREEN_LED, HIGH);
//   digitalWrite(RED_LED, LOW);
//   digitalWrite(MOTOR_IN1, HIGH);
//   digitalWrite(MOTOR_IN2, LOW);
//   delay(3000);
  
//   Serial.println("Motor STOP");
//   digitalWrite(GREEN_LED, LOW);
//   digitalWrite(MOTOR_IN1, LOW);
//   digitalWrite(MOTOR_IN2, LOW);
//   delay(2000);
  
//   Serial.println("Motor REVERSE - Red LED");
//   digitalWrite(RED_LED, HIGH);
//   digitalWrite(GREEN_LED, LOW);
//   digitalWrite(MOTOR_IN1, LOW);
//   digitalWrite(MOTOR_IN2, HIGH);
//   delay(3000);
  
//   Serial.println("Motor STOP");
//   digitalWrite(RED_LED, LOW);
//   digitalWrite(MOTOR_IN1, LOW);
//   digitalWrite(MOTOR_IN2, LOW);
//   delay(2000);
// }