#include "Servo.h"

// compiler pre-processor directive
#define BAUD_RATE 9600
#define THROTTLE_IN_PIN 3
#define STEERING_IN_PIN 4
#define THROTTLE_OUT_PIN 5
#define STEERING_OUT_PIN 6
#define NEUTRAL_THROTTLE 1500
#define NEUTRAL_STEERING 1500


// never changing constants
const char HANDSHAKE_START = '(';
const char HANDSHAKE_END = ')';
const byte RC_CONTROLLER_STATE_PIN = 2;
const int PULSEWIDTH = 10000;
const unsigned int MAX_DISCONNECT_COUNT = 10;
const long disconnectedDurationBuffer = 1000; // milisecond
const unsigned long disconnectedDurationSentinel= 2000; // 5 seconds


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
unsigned long prevBluetoothConnectedTime = 0;
unsigned long prevJetsonConnectedTime = 0;
String receivedData = "";
bool isForwardState = true; // car is currently in forward state.

bool is_controller_values_changing_now = false;

// volatile variables that only changes in interrupts
volatile int32_t controller_throttle_read = 0;
volatile int32_t controller_steering_read = 0;
volatile int32_t prev_controller_throttle_read = 0;
volatile int32_t prev_controller_steering_read = 0;

// Servo Declaration
Servo servoThrottle;
Servo servoSteering;

void setup() {
  // set pins
  pinMode(RC_CONTROLLER_STATE_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  // attach interrupts
  attachInterrupt(digitalPinToInterrupt(THROTTLE_IN_PIN), onThrottleChanged, CHANGE);
  attachInterrupt(digitalPinToInterrupt(STEERING_IN_PIN), onSteeringChanged, CHANGE);

  // begin serial & Servo
  Serial.begin(BAUD_RATE);
  Serial1.begin(BAUD_RATE); // bluetooth on PIN 14, 15
  servoThrottle.attach(THROTTLE_OUT_PIN);
  servoSteering.attach(STEERING_OUT_PIN);

}

void loop() {
  // detect PINs
  detectControllerState();
  
  // determine state from detection
  rc_controller_state = determineTraxxasControllerConnected();//determineTraxxasControllerConnected(); // true if connected, false otherwise ( 0 == true, 1 == false)

  bool jetson_controller_state = determineJetsonControllerConnected();  
  
  isControllerConnected = rc_controller_state or jetson_controller_state;
  if (isControllerConnected == false) {
    blinkLED();
  } else {
    turnOnLED();
  } 

  if (rc_controller_state == true) {
      latest_throttle = controller_throttle_read;
      latest_steering = controller_steering_read;
  } else if (jetson_controller_state == true) {
      parseData();
      writeToSerial(latest_throttle, latest_steering);
  }

  ensureSmoothBackTransition();
  writeToServo(latest_throttle, latest_steering);
}


bool detectControllerState2() {
  /*
   * if controller throttle and steering has not been changing (~5 pwm) for 5 secods --> controller is disconnected
   * else, controller is connected
   * 
   */
  detectControllerState2Helper();
  if (is_controller_values_changing_now) {
    return true;
  } else if (millis() - prevControllerConnectedTime < disconnectedDurationSentinel) {
    return true;
  } else {
    // if controller hasnt been sending new cmds in disconnectedDurationSentinel seconds
    if (1480 < controller_throttle_read and controller_throttle_read < 1520 and 1480 < controller_steering_read and controller_steering_read < 1520) {
      // and if the current throttle and steering read is in neutral
      // i assume that the user does not want to use the controller
      return false;
    } else {
      // if the throttle and steering is not in neutral
      // i should still obey the order
      return true;
    }

  }
}

void detectControllerState2Helper() {
  if (abs(prev_controller_throttle_read - controller_throttle_read) > 5 or abs(prev_controller_steering_read - controller_steering_read) > 5) {
    is_controller_values_changing_now = true;
    prevControllerConnectedTime = millis();
  } else {
    is_controller_values_changing_now = false;
  }
}



void ensureSmoothBackTransition() {
  if (isForwardState and latest_throttle < 1495) {
    for (int i = 0; i <= 5; i++) {
      writeToServo(1000, latest_steering);
    }
    for (int i = 0; i <= 10; i++) {
      writeToServo(1500,latest_steering);
    }
    isForwardState = false;
  } else if (latest_throttle > 1505) {
    isForwardState = true;
  }
}


bool isBluetoothConnected() { 
  /*
   * Return true if bluetooth is connected, false otherwise
   */
    if (Serial1.available() > 0) {
      prevBluetoothConnectedTime = millis();
      return true;
    } else {
      // bluetooth will have delay so make sure to put some buffer time to determine whether it is actually disconnected
      unsigned long currentMillis = millis();
      if (currentMillis - prevBluetoothConnectedTime > disconnectedDurationBuffer) {
        return false;
      }
      return true;
    }
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

void checkServo() {
  if (servoThrottle.attached() == false) {
    servoThrottle.attach(THROTTLE_OUT_PIN);
  }
  if (servoSteering.attached() == false) {
    servoSteering.attach(STEERING_OUT_PIN);
  }
}

void parseData() {
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

bool determineTraxxasControllerConnected() {
  // return true if controller is connected, false otherwise
  if (rc_controller_disconnect_count < MAX_DISCONNECT_COUNT) {
    return true ;
  }
  return false;
}

void writeToServo(unsigned int throttle, unsigned int steering) {
  // prevent servo from detaching
  checkServo();
  latest_throttle = throttle;
  latest_steering = steering;
  servoThrottle.writeMicroseconds(latest_throttle);
  servoSteering.writeMicroseconds(latest_steering);
//  writeToSerial(latest_throttle, latest_steering);
}

void writeToBluetooth(unsigned int throttle, unsigned int steering) {
  /*
   * Write to Bluetooth Device
   */
  Serial1.print(HANDSHAKE_START);
  Serial1.print(throttle);
  Serial1.print(",");
  Serial1.print(steering);
  Serial1.println(HANDSHAKE_END);
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
void onThrottleChanged(){
  /*
   * Parse Controller Throttle value
   */
  static uint32_t ulStart;
  if (digitalRead(THROTTLE_IN_PIN)){
    ulStart = micros();
  } else{
    prev_controller_throttle_read = controller_throttle_read;
    controller_throttle_read = (uint32_t)(micros() - ulStart);
  }
}
void onSteeringChanged(){
  /*
   * Parse Controller Steering value
   */
  static uint32_t ulStart;
  if (digitalRead(STEERING_IN_PIN)){
    ulStart = micros();
  } else{
    prev_controller_steering_read = controller_steering_read;
    controller_steering_read = (uint32_t)(micros() - ulStart);
  }
}
void detectControllerState() {
  /*
   * Parse Controller State
   */
  if (pulseIn(RC_CONTROLLER_STATE_PIN, HIGH, PULSEWIDTH) > 0) {
    rc_controller_disconnect_count = 0;
  } else{
    rc_controller_disconnect_count += 1;
  }
}
