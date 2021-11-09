#include "Servo.h"
#include "ArduinoBLE.h"

#define BAUD_RATE 9600
#define THROTTLE_IN_PIN 3
#define STEERING_IN_PIN 4
#define THROTTLE_OUT_PIN 5
#define STEERING_OUT_PIN 6
#define NEUTRAL_THROTTLE 1500
#define NEUTRAL_STEERING 1500
#define BUFFER_SIZE 10
#define MAX_THROTTLE 2000
#define MIN_THROTTLE 1000
#define MAX_STEERING 1000
#define MIN_STEERING 2000
#define THROTTLE_FLAG 1
#define STEERING_FLAG 2
#define AUX_FLAG 4

// never changing constants
const char HANDSHAKE_START = '(';
const char HANDSHAKE_END = ')';
const int PULSEWIDTH = 10000;
const unsigned int MAX_DISCONNECT_COUNT = 10;
const long disconnectedDurationBuffer = 1000; // milisecond
const unsigned long disconnectedDurationSentinel= 100; // 10 seconds
const int cum = 10;


// state variables that might change
bool isControllerConnected = false;
bool rc_controller_state = false; // by default, controller is not connected
unsigned int rc_controller_disconnect_count = 0;
unsigned int latest_throttle = NEUTRAL_THROTTLE; // set to neutral by default
unsigned int latest_steering = NEUTRAL_STEERING; // set to neutral by default
int ledState = LOW;
const long ledInterval = 500; // milisecond
unsigned long prevControllerConnectedTime = 0;
unsigned long previousMillis = 0;  // will store last time LED was updated
unsigned long prevJetsonConnectedTime = 0;
String receivedData = "";
bool isForwardState = true; // car is currently in forward state.
bool is_bluetooth_connected = false;
bool is_controller_values_changing_now = false;

volatile uint8_t bUpdateFlagsShared;
volatile uint16_t unThrottleInShared;
volatile uint16_t unSteeringInShared;

volatile int32_t bluetooth_throttle_read = 1500;
volatile int32_t bluetooth_steering_read = 1500;
volatile int32_t controller_throttle_read = 1500;
volatile int32_t controller_steering_read = 1500;
volatile int32_t prev_controller_throttle_read = 1500;
volatile int32_t prev_controller_steering_read = 1500;
volatile unsigned long steering_start_period = 0;
volatile boolean new_steering_signal = false;


uint32_t ulThrottleStart;
uint32_t ulSteeringStart;
// bluetooth device, look at https://ladvien.com/arduino-nano-33-bluetooth-low-energy-setup/
BLEService vehicleControlService("19B10010-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic vehicleControlCharacteristic("19B10011-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse, 11, true);


Servo servoThrottle;
Servo servoSteering;

void setup()
{
  Serial.begin(BAUD_RATE);
  pinMode(LED_BUILTIN, OUTPUT);
  servoThrottle.attach(THROTTLE_OUT_PIN);
  servoSteering.attach(STEERING_OUT_PIN);

  attachInterrupt(THROTTLE_IN_PIN, calcThrottle,CHANGE);
  attachInterrupt(STEERING_IN_PIN, calcSteering,CHANGE);

    // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("IR Vehicle Control Unit");
  BLE.setAdvertisedService(vehicleControlService);
  vehicleControlService.addCharacteristic(vehicleControlCharacteristic);
  BLE.addService(vehicleControlService);

  // assign event handlers for connected, disconnected to peripheral
  BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);

  // assign event handlers for characteristic
  vehicleControlCharacteristic.setEventHandler(BLEWritten, vehicleControllerCharacteristicWritten);

  BLE.advertise();
  Serial.println("Bluetooth device active, waiting for connections...");
}

void loop()
{
  checkServo();
  BLE.poll();
  read_controller();
  rc_controller_state = false;//detectControllerState2();
  bool jetson_controller_state = determineJetsonControllerConnected();
  bool bluetooth_controller_state = is_bluetooth_connected;

  isControllerConnected = rc_controller_state or jetson_controller_state or bluetooth_controller_state;
  if (isControllerConnected == false) {
    blinkLED();
  } else {
    turnOnLED();
  }
  Serial.print(rc_controller_state);
  Serial.print(jetson_controller_state);
  Serial.print(bluetooth_controller_state);

  if (rc_controller_state == true) {
//      uint16_t t = read_rc_controller(THROTTLE_IN_PIN);
      latest_throttle = controller_throttle_read;
      latest_steering = controller_steering_read;
  } else if (jetson_controller_state == true) {
      parseSerialData();
  } else if (bluetooth_controller_state == true) {
      latest_throttle = bluetooth_throttle_read; // done in vehicleControllerCharacteristicWritten call back handler
      latest_steering = bluetooth_steering_read; // done in vehicleControllerCharacteristicWritten call back handler
  } else {
    latest_throttle = NEUTRAL_THROTTLE;
    latest_steering = NEUTRAL_STEERING;
  }

  ensureSmoothBackTransition();
  writeToServo(latest_throttle, latest_steering);
}

uint16_t read_rc_controller(int pin){
  int i = (uint16_t) pulseIn(pin, HIGH);
  return (uint16_t) ((i+5) / 10) * 10; // rounding for more stablization
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

void turnOnLED() {
  digitalWrite(LED_BUILTIN, HIGH);
}

void blinkLED() {

  // code for blinking led
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= ledInterval) {
    // save the last time you blinked the LED
    previousMillis = currentMillis;

    // if the LED is off turn it on and vice-versa:
    if (ledState == LOW) {
      ledState = HIGH;

    } else {
      ledState = LOW;
    }

    // set the LED with the ledState of the variable:
    digitalWrite(LED_BUILTIN, ledState);
  }
}

void parseSerialData() {
  /*
      method to parse received data given input method.
      Currently supported channel are:
        Serial
        Serial1
      Messages are required to be in the format of HANDSHAKE_START PWMTHROTTLE STEERING HANDSHAKE_END
      For example, by default, HANDSHAKE_START= "(", HANDSHAKE_END=")"
      Then a sample receivedData could be (1500,1500)
  */
    do {
        char buf[20];
        size_t num_read = Serial.readBytesUntil(HANDSHAKE_END, buf, 20);
        char *token = strtok(buf, ",");
        if (token[0] == HANDSHAKE_START) {
          if (token != NULL) {
            unsigned int curr_throttle_read = atoi(token + 1);
            if (curr_throttle_read >= 1000 and curr_throttle_read <= 2000) {
              latest_throttle = curr_throttle_read;
            }
          }
          token = strtok(NULL, ",");
          if (token != NULL) {
            unsigned int curr_steering_read = atoi(token);
            if (curr_steering_read >= 1000 and curr_steering_read <= 2000) {
              latest_steering = curr_steering_read;
            }
          }
        }
    } while(Serial.available() > 0);
}
bool determineJetsonControllerConnected() {
  /*
   * Return true if jetson is connected, false otherwise
   */
    if (Serial.available() > 0) {
      prevJetsonConnectedTime = millis();
      return true;
    } else {
      // bluetooth will have delay so make sure to put some buffer time to determine whether it is actually disconnected
      unsigned long currentMillis = millis();
      if (currentMillis - prevJetsonConnectedTime > disconnectedDurationBuffer) {
        return false;
      }
      return true;
    }
}

bool detectControllerState2() {
  /*
   * if controller throttle and steering has not been changing (~5 pwm) for 5 secods --> controller is disconnected
   * else, controller is connected
   *
   */
  detectControllerState2Helper();
  if (is_controller_values_changing_now) {
//    Serial.print("controller changing, return true");
    return true;
  } else if (millis() - prevControllerConnectedTime < disconnectedDurationSentinel) {
//    Serial.print("not time yet, return true");
    return true;
  } else {
    // if controller hasnt been sending new cmds in disconnectedDurationSentinel seconds
    if (1400 < controller_throttle_read and controller_throttle_read < 1600 and 1400 < controller_steering_read and controller_steering_read < 1600) {
      // and if the current throttle and steering read is in neutral
      // i assume that the user does not want to use the controller
//      Serial.print("within range, return false");
      return false;
    } else {
      // if the throttle and steering is not in neutral
      // i should still obey the order
//      Serial.print("outside range, return true");
      return true;
    }
  }
}

void detectControllerState2Helper() {
  if (abs(prev_controller_throttle_read - controller_throttle_read) > 300 or abs(prev_controller_steering_read - controller_steering_read) > 300) {
    is_controller_values_changing_now = true;
    prevControllerConnectedTime = millis();
  } else {
    is_controller_values_changing_now = false;
  }
}

void checkServo() {
  if (servoThrottle.attached() == false) {
    servoThrottle.attach(THROTTLE_OUT_PIN);
  }
  if (servoSteering.attached() == false) {
    servoSteering.attach(STEERING_OUT_PIN);
  }
}

void writeToServo(unsigned int throttle, unsigned int steering) {
  // prevent servo from detaching
  checkServo();
  latest_throttle = throttle;
  latest_steering = steering;

  servoThrottle.writeMicroseconds(latest_throttle);
  servoSteering.writeMicroseconds(latest_steering);
  writeToSerial(latest_throttle, latest_steering);
}

void writeToSerial(unsigned int throttle, unsigned int steering) {
  /*
   * Write to Serial
   */
  Serial.print(HANDSHAKE_START);
  Serial.print(throttle);
  Serial.print(",");
  Serial.print(steering);
  Serial.println(HANDSHAKE_END);
}

void read_controller() {
  static uint16_t unThrottleIn;
  static uint16_t unSteeringIn;
  static uint8_t bUpdateFlags;

  if(bUpdateFlagsShared)
  {
    noInterrupts();
    bUpdateFlags = bUpdateFlagsShared;
    if(bUpdateFlags & THROTTLE_FLAG)
    {
      unThrottleIn = unThrottleInShared;
    }
    if(bUpdateFlags & STEERING_FLAG)
    {
      unSteeringIn = unSteeringInShared;
    }
    bUpdateFlagsShared = 0;
    interrupts();
  }

  if(bUpdateFlags & THROTTLE_FLAG)
  {
    if(servoThrottle.readMicroseconds() != unThrottleIn)
    {
      prev_controller_throttle_read = controller_throttle_read;
      controller_throttle_read = (unThrottleIn + prev_controller_throttle_read * cum) / (cum+1);
    }
  }

  if(bUpdateFlags & STEERING_FLAG)
  {
    if(servoSteering.readMicroseconds() != unSteeringIn)
    {
      prev_controller_steering_read = controller_steering_read;
      controller_steering_read = (unSteeringIn + prev_controller_steering_read * cum) / (cum+1);
//      Serial.println(controller_steering_read);
    } else {
//      Serial.println("HERE");
    }
  }


  bUpdateFlags = 0;
}

void calcThrottle()
{
  if(digitalRead(THROTTLE_IN_PIN) == HIGH)
  {
    ulThrottleStart = micros();
  }
  else
  {
    unThrottleInShared = (uint16_t)(micros() - ulThrottleStart);
    bUpdateFlagsShared |= THROTTLE_FLAG;
  }
}

void calcSteering()
{
  if(digitalRead(STEERING_IN_PIN) == HIGH)
  {
    ulSteeringStart = micros();
  }
  else
  {
    unSteeringInShared = (uint16_t)(micros() - ulSteeringStart);
    bUpdateFlagsShared |= STEERING_FLAG;
  }
}


void blePeripheralConnectHandler(BLEDevice central) {
  // central connected event handler
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
  is_bluetooth_connected = true;
  latest_throttle = 1500;
  latest_steering = 1500;
  bluetooth_throttle_read = 1500;
  bluetooth_steering_read = 1500;
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  Serial.print("Disconnected event, central: ");
  Serial.println(central.address());
  is_bluetooth_connected = false;
  latest_throttle = 1500;
  latest_steering = 1500;
  bluetooth_throttle_read = 1500;
  bluetooth_steering_read = 1500;
}

void vehicleControllerCharacteristicWritten(BLEDevice central, BLECharacteristic characteristic) {
  // central wrote new value to characteristic, update LED
  is_bluetooth_connected = true;
  char buf[10];
  characteristic.readValue(buf, 10);
  char *token = strtok(buf, ",");
  if (token[0] == HANDSHAKE_START) {
    if (token != NULL) {
      unsigned int curr_throttle_read = atoi(token + 1);
      if (curr_throttle_read >= 1000 and curr_throttle_read <= 2000) {
        bluetooth_throttle_read = curr_throttle_read;
      }
    }
    token = strtok(NULL, ",");
    if (token != NULL) {
      unsigned int curr_steering_read = atoi(token);
      if (curr_steering_read >= 1000 and curr_steering_read <= 2000) {
        bluetooth_steering_read = curr_steering_read;
      }
    }
  }
}
