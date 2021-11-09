#include <string.h>
#include <Servo.h>

int throttleIn = 1500; //global variable for throttle
int steeringIn = 1500; //global variable for steering

const int MASTER_MODE = 0;
const int SLAVE_MODE = 1;

int mode = MASTER_MODE;

class Car {
  public:
    void initArduino();
    void readReceiver();
    void writeToActuators();
    float saturateMotor(float x);
    float saturateServo(float x);

  private:

    const int MOTOR_PIN = 11;
    const int SERVO_PIN = 10;

    const int ch1 = 4;
    const int ch2 = 5;

    Servo throttle;
    Servo steering;

    const int MOTOR_MAX = 1750;
    const int MOTOR_MIN = 800;
    const int MOTOR_NEUTRAL = 1500;
    const int THETA_CENTER = 1500;
    const int THETA_MAX = 3000;
    const int THETA_MIN = 0;

    uint16_t microseconds2PWM(uint16_t microseconds);
};

Car car;

void setup() {
  Serial.begin(9600);
  car.initArduino();
}

//Main Program
void loop() {
  if (mode == MASTER_MODE) {
    car.readReceiver();
    car.writeToActuators();
    Serial.print(car.saturateMotor(throttleIn));
    Serial.print(" ");
    Serial.print(car.saturateServo(steeringIn));
    Serial.println();
  }
  else if (mode == SLAVE_MODE) {
    char input[100];
    byte size = Serial.readBytes(input, 100);
    input[size] = 0;
    char *s = strtok(input, " ");
    if (s != NULL) throttleIn = atoi(s);
    s = strtok(NULL, " ");
    if (s != NULL) steeringIn = atoi(s);
    car.writeToActuators();
  }
}

void Car::initArduino() {
  throttle.attach(MOTOR_PIN);
  steering.attach(SERVO_PIN);
  pinMode(ch1, INPUT);
  pinMode(ch2, INPUT);
}

void Car::readReceiver() {
  throttleIn = pulseIn (ch2, HIGH);
  steeringIn = pulseIn (ch1, HIGH);
}

float Car::saturateMotor(float x) {
  if (x > MOTOR_MAX) { x = MOTOR_MAX; }
  if (x < MOTOR_MIN) { x = MOTOR_MIN; }
  return x;
}

float Car::saturateServo(float x) {
  if (x > THETA_MAX) { x = THETA_MAX; }
  if (x < THETA_MIN) { x = THETA_MIN; }
  return x;
}

void Car::writeToActuators() {
  throttle.writeMicroseconds( (uint16_t) saturateMotor( throttleIn ) );
  steering.writeMicroseconds( (uint16_t) saturateServo( steeringIn ) );
}
