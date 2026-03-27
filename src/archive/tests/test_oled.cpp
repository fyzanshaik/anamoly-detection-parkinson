// // OLED Test - Gateway Component Test 1
// // Upload this FIRST to verify OLED is working

// #include <Arduino.h>
// #include <Wire.h>
// #include <Adafruit_GFX.h>
// #include <Adafruit_SSD1306.h>

// // OLED Display settings
// #define SCREEN_WIDTH 128
// #define SCREEN_HEIGHT 64  // Change to 32 if you have smaller OLED
// #define OLED_RESET -1     // No reset pin
// #define SCREEN_ADDRESS 0x3C  // Common I2C address, try 0x3D if this fails

// Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// void setup() {
//   Serial.begin(115200);
//   delay(2000);

//   Serial.println("=== Gateway OLED Test ===");

//   // Initialize I2C with SDA=21, SCL=22
//   Wire.begin(21, 22);

//   // Try to initialize display
//   if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
//     Serial.println("ERROR: OLED not found!");
//     Serial.println("Trying alternate address 0x3D...");

//     if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
//       Serial.println("ERROR: OLED still not found at 0x3D!");
//       Serial.println("Check connections:");
//       Serial.println("  VCC → 3V3");
//       Serial.println("  GND → GND");
//       Serial.println("  SDA → GPIO 21");
//       Serial.println("  SCL → GPIO 22");
//       while(1);  // Halt
//     } else {
//       Serial.println("SUCCESS: OLED found at 0x3D");
//     }
//   } else {
//     Serial.println("SUCCESS: OLED found at 0x3C");
//   }

//   // Clear display
//   display.clearDisplay();

//   // Test pattern
//   display.setTextSize(1);
//   display.setTextColor(SSD1306_WHITE);
//   display.setCursor(0, 0);
//   display.println("Gateway Node");
//   display.println("OLED Test");
//   display.println("---------------");
//   display.setTextSize(2);
//   display.println("SUCCESS!");

//   display.display();
//   Serial.println("OLED displaying test pattern");
// }

// void loop() {
//   // Blink counter on OLED
//   static int counter = 0;

//   delay(1000);
//   counter++;

//   display.clearDisplay();
//   display.setTextSize(1);
//   display.setCursor(0, 0);
//   display.println("Gateway OLED OK");
//   display.println("---------------");
//   display.setTextSize(2);
//   display.print("Count: ");
//   display.println(counter);
//   display.display();

//   Serial.print("OLED update: ");
//   Serial.println(counter);
// }
