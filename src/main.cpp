//////////////////////////////////////////////////////////////////////
////////////// home automation by Martin Mittrenga ///////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <TFT_eSPI.h>           /* Please use the TFT library provided in the library. */

#define BUTTON_PIN                    35

#define BMI160_I2C_ADDRESS 0x68
#define ACCEL_SENSITIVITY 16384.0f  // ±2g

bool buttonState = false;
unsigned long lastButtonChanged = 0;
unsigned long lastTurnOff = 0;
unsigned long lastDisplayPart = 0;

int step = 1;
int waitShort = 200;
int waitLong = 300;
int debounceTime = 200;
int displayOffTime = 300000;

char buf_a[16];
char buf_accel_x[16];

//////////////////////////////////////////////////////////////////////

TFT_eSPI tft = TFT_eSPI();

//////////////////////////////////////////////////////////////////////

// Funktionsdeklaration
void autoCalibrateAccelerometer();

//////////////////////////////////////////////////////////////////////

void sleepRequest() {
  if (millis() - lastButtonChanged > debounceTime) {
    buttonState = digitalRead(BUTTON_PIN);
    lastButtonChanged = millis(); 

    if (buttonState == false) {

      tft.fillScreen(TFT_BLACK);
      Serial.println("Going to sleep now because of button push");
      esp_deep_sleep_start();
    }
  }
}

//////////////////////////////////////////////////////////////////////

void setup()
{
    // Serial communication
    Serial.begin(115200);
    delay(100);

    esp_sleep_enable_ext0_wakeup(GPIO_NUM_0,0); //1 = true, 0 = false

    pinMode(BUTTON_PIN, INPUT);

    //tft.init();
    tft.begin();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
    //tft.setSwapBytes(true);
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE);

    lastTurnOff = millis();

    Serial.println();
    Serial.println("Home Automation");
    tft.fillScreen(TFT_BLACK);
    tft.drawString("Home Automation", 20, 20, 4); 
    tft.drawString("", 20, 50, 4); 
    delay(1000);

    Serial.println();
    Serial.println("Accelerometer");
    Serial.println("Gyroskop");
    tft.fillScreen(TFT_BLACK);
    tft.drawString("Accelerometer", 20, 20, 4); 
    tft.drawString("Gyroskop", 20, 50, 4); 
    delay(1000);

    // I2C communication
    Wire.begin();

    // Initialize BMI160 accelerometer
    Wire.beginTransmission(BMI160_I2C_ADDRESS);
    Wire.write(0x7E);     // Command register
    Wire.write(0x11);     // Accelerometer normal mode
    Wire.endTransmission();

    delay(100);

    // Perform accelerometer auto-calibration
    autoCalibrateAccelerometer();

    Serial.println("BMI160 Initialized and Calibrated");
    Serial.println();

    tft.fillScreen(TFT_BLACK);
    tft.drawString("BMI160 init", 20, 20, 4); 
    tft.drawString("", 20, 50, 4); 
    delay(1000);
}

void loop()
{

    if ((millis() - lastDisplayPart > waitShort) && (step == 1)) {

      int16_t ax = 0;
      int16_t ay = 0;
      int16_t az = 0;

      // --------------------------------------------------
      // Read accelerometer data
      // --------------------------------------------------

      Wire.beginTransmission(BMI160_I2C_ADDRESS);
      Wire.write(0x12);     // Start register for accelerometer data
      Wire.endTransmission(false);

      Wire.requestFrom(BMI160_I2C_ADDRESS, 6);

      if (Wire.available() == 6)
      {
          ax = (int16_t)(Wire.read() | (Wire.read() << 8));
          ay = (int16_t)(Wire.read() | (Wire.read() << 8));
          az = (int16_t)(Wire.read() | (Wire.read() << 8));
      }

      // --------------------------------------------------
      // Convert raw values to g
      // --------------------------------------------------

      float ax_g = ax / ACCEL_SENSITIVITY;
      float ay_g = ay / ACCEL_SENSITIVITY;
      float az_g = az / ACCEL_SENSITIVITY;

      // --------------------------------------------------
      // Convert raw values to m/s²
      // --------------------------------------------------

      float ax_mps2 = ax_g * 9.81f;
      float ay_mps2 = ay_g * 9.81f;
      float az_mps2 = az_g * 9.81f;

      // --------------------------------------------------
      // Calculate pitch and roll
      // --------------------------------------------------

      float pitch =
          atan2(
              ay_g,
              sqrt(ax_g * ax_g + az_g * az_g)
          )
          * 180.0f / PI;

      float roll =
          atan2(
              -ax_g,
              az_g
          )
          * 180.0f / PI;

      // --------------------------------------------------
      // Serial output
      // --------------------------------------------------

      snprintf(buf_accel_x, sizeof(buf_a), "%.2f", ax_mps2);

      tft.fillScreen(TFT_BLACK);
      tft.drawString("Accel (m/s^2): ", 20, 20, 4); 
      tft.drawString(buf_accel_x, 20, 50, 4);

      Serial.print("Accel (m/s^2): ");
      Serial.print(ax_mps2, 2);
      Serial.print(", ");
      Serial.print(ay_mps2, 2);
      Serial.print(", ");
      Serial.println(az_mps2, 2);

      Serial.print("Accel (g):     ");
      Serial.print(ax_g, 3);
      Serial.print(", ");
      Serial.print(ay_g, 3);
      Serial.print(", ");
      Serial.println(az_g, 3);

      Serial.print("Pitch: ");
      Serial.print(pitch, 2);
      Serial.print(" deg");

      Serial.print(" | Roll: ");
      Serial.print(roll, 2);
      Serial.println(" deg");

      Serial.println("--------------------------------");
      
      step = 1;
      lastDisplayPart = millis();
    }

    // Read Button and do anything
    if (millis() - lastButtonChanged > debounceTime) {
      buttonState = digitalRead(BUTTON_PIN);
      lastButtonChanged = millis(); 

      if (buttonState == false) {

        tft.fillScreen(TFT_BLACK);
        Serial.println("Going to sleep now because of button push");
        esp_deep_sleep_start();
      }
    }

    // Function turn off the Display after some time
    if ((millis() - lastTurnOff > displayOffTime)) {

      tft.fillScreen(TFT_BLACK);
      Serial.println("Going to sleep now because of timer");
      esp_deep_sleep_start();
    }

}

// ======================================================
// Accelerometer Auto Calibration
// ======================================================

void autoCalibrateAccelerometer()
{
    Serial.println("Starting accelerometer auto-calibration...");
    // Configure accelerometer for auto-calibration
    Wire.beginTransmission(BMI160_I2C_ADDRESS);
    Wire.write(0x7E);     // Command register
    Wire.write(0x37);     // Start accelerometer offset calibration
    Wire.endTransmission();
    delay(100);
    // Wait for calibration to complete
    delay(1000);

    Serial.println("Accelerometer Auto-Calibration Complete");
    tft.fillScreen(TFT_BLACK);
    tft.drawString("BMI160 init", 20, 20, 4); 
    tft.drawString("", 20, 50, 4); 
    delay(1000);
}