#define RED_LED_PIN 33

const unsigned long redLEDToggleDuration = 500; // ms


bool isRedLEDOn = false;
unsigned long lastREDLEDToggleTime = 0;  // will store last time LED was updated

const int ledPin = 2;


void setup() {
  Serial.begin(115200);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(ledPin, OUTPUT);

}
void loop() {
  blinkRedLED();
  Serial.println("Looping");
}


void blinkRedLED() {
  unsigned long currentMillis = millis();
  if (isRedLEDOn && (currentMillis - lastREDLEDToggleTime >= redLEDToggleDuration)) {
    digitalWrite(RED_LED_PIN, HIGH);
      digitalWrite(ledPin, HIGH);

    isRedLEDOn = false;
    lastREDLEDToggleTime = currentMillis;
  } else if (isRedLEDOn == false && (currentMillis -lastREDLEDToggleTime >= redLEDToggleDuration)) {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(ledPin, LOW);

    isRedLEDOn = true;
    lastREDLEDToggleTime = currentMillis;
  }
}
