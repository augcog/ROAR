#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <ESP32Servo.h>

#define SERVICE_UUID        "19B10010-E8F2-537E-4F6C-D104768A1214"
#define CHARACTERISTIC_UUID "19B10011-E8F2-537E-4F6C-D104768A1214"


#define THROTTLE_PIN 14
#define STEERING_PIN 2
#define FLASH_LED_PIN 4
#define RED_LED_PIN 33

const char HANDSHAKE_START = '(';
const unsigned long redLEDToggleDuration = 500; // ms



volatile int32_t ws_throttle_read = 1500;
volatile int32_t ws_steering_read = 1500;
bool isForwardState = true; // car is currently in forward state.
unsigned int latest_throttle = 1500; // set to neutral by default
unsigned int latest_steering = 1500; // set to neutral by default

Servo throttleServo;
Servo steeringServo;


bool isFlashLightOn = false;
bool isRedLEDOn = false;
unsigned long lastREDLEDToggleTime = 0;  // will store last time LED was updated

bool deviceConnected = false;



class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      std::string value = pCharacteristic->getValue();
      if (value.length() > 0) {
        String argument = String(value.c_str());
        char buf[value.length()] = "\0";
        argument.toCharArray(buf, argument.length());
        char *token = strtok(buf, ",");
        if (token[0] == HANDSHAKE_START) {
            if (token != NULL) {
              unsigned int curr_throttle_read = atoi(token + 1);
              if (curr_throttle_read >= 1000 and curr_throttle_read <= 2000) {
                ws_throttle_read = curr_throttle_read;
              } 
            }
            token = strtok(NULL, ",");
            if (token != NULL) {
              unsigned int curr_steering_read = atoi(token);
              if (curr_steering_read >= 1000 and curr_steering_read <= 2000) {
                ws_steering_read = curr_steering_read;
              }
            }
        }
//
//        Serial.print(ws_throttle_read);
//        Serial.print(",");
//        Serial.println(ws_steering_read);
        
      }
    }

    void onRead(BLECharacteristic *pCharacteristic, esp_ble_gatts_cb_param_t* param){
      pCharacteristic->setValue("hello world");
    }
};


class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Connected");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Disconnected");
      BLEDevice::startAdvertising();
    }
};

void setup() {
  Serial.begin(115200);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(FLASH_LED_PIN, OUTPUT);


  setupServo();
  setupBLE();

}

void loop() {
  if (deviceConnected == false) {
    blinkRedLED();
  } else {
    digitalWrite(RED_LED_PIN, LOW);
  }

  
  ensureSmoothBackTransition();
  writeToServo(ws_throttle_read, ws_steering_read);

}


void setupBLE() {
    BLEDevice::init("IR Vehicle Control");
    BLEServer *pServer = BLEDevice::createServer();
  
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pServer->setCallbacks(new MyServerCallbacks());
    
    BLECharacteristic *pCharacteristic = pService->createCharacteristic(
                                           CHARACTERISTIC_UUID,
                                           BLECharacteristic::PROPERTY_READ |
                                           BLECharacteristic::PROPERTY_WRITE | 
                                           BLECharacteristic::PROPERTY_WRITE_NR
                                         );

    pCharacteristic->setCallbacks(new MyCallbacks());
    pService->start();
  
//    BLEAdvertising *pAdvertising = pServer->getAdvertising();
//    pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
//    pAdvertising->setMinPreferred(0x12);
//    pAdvertising->start();

    BLEDevice::startAdvertising();

    Serial.println("BLE Device Started");
}

void setupServo() {
  ESP32PWM::timerCount[0]=4;
  ESP32PWM::timerCount[1]=4;
  throttleServo.setPeriodHertz(50);
  throttleServo.attach(THROTTLE_PIN, 1000, 2000);
  steeringServo.setPeriodHertz(50);    // standard 50 hz servo
  steeringServo.attach(STEERING_PIN, 1000, 2000); // attaches the servo on pin (whatever you assign)
}

void writeToServo(unsigned int throttle, unsigned int steering) {
  checkServo(); // prevent servo from detaching
  latest_throttle = throttle;
  latest_steering = steering;

  throttleServo.writeMicroseconds(latest_throttle);
  steeringServo.writeMicroseconds(latest_steering);
  writeToSerial(latest_throttle, latest_steering);
}

void writeToSerial(unsigned int throttle, unsigned int steering) {
  /*
   * Write to Serial
   */
  Serial.print(throttle);
  Serial.print(",");
  Serial.println(steering);
}
void checkServo() {
  if (throttleServo.attached() == false) {
    throttleServo.attach(THROTTLE_PIN);
  }
  if (steeringServo.attached() == false) {
    steeringServo.attach(STEERING_PIN);
  }
}

void ensureSmoothBackTransition() {
  if (isForwardState and latest_throttle < 1500) {
    writeToServo(1500, latest_steering);
    delay(100);
    writeToServo(1450, latest_steering);
    delay(100);
    writeToServo(1500,latest_steering);
    delay(100);
    isForwardState = false;
  } else if (latest_throttle >= 1500) {
    isForwardState = true;
  }
}


// Utility functions
void blinkFlashlight() {
  if (isFlashLightOn) {
    ledcWrite(FLASH_LED_PIN, 0);
    isFlashLightOn = false;
  } else {
    ledcWrite(FLASH_LED_PIN, 100);
    isFlashLightOn = true;
  }
}


void blinkRedLED() {
  unsigned long currentMillis = millis();
  if (isRedLEDOn && (currentMillis - lastREDLEDToggleTime >= redLEDToggleDuration)) {
    digitalWrite(RED_LED_PIN, HIGH);
    isRedLEDOn = false;
    lastREDLEDToggleTime = currentMillis;
  } else if (isRedLEDOn == false && (currentMillis -lastREDLEDToggleTime >= redLEDToggleDuration)) {
    digitalWrite(RED_LED_PIN, LOW);
    isRedLEDOn = true;
    lastREDLEDToggleTime = currentMillis;
  }
}
