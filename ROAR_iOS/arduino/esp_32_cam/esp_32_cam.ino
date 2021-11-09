#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>
#include <uri/UriBraces.h>
#include <ESP32Servo.h>
#include <ESPAsyncWebServer.h>
#include <AsyncTCP.h>
#include "esp_camera.h"


#define THROTTLE_PIN 14
#define STEERING_PIN 2
#define FLASH_LED_PIN 4
#define RED_LED_PIN 33


// CHANGE YOUR WIFI CREDENTIALS!
char* WIFI_SSID = "NETGEAR78";
char* WIFI_PASS = "wuxiaohua1011";
const uint8_t fps = 10;    //sets minimum delay between frames, HW limits of ESP32 allows about 12fps @ 800x600


static auto loRes = esp32cam::Resolution::find(320, 240);
const char HANDSHAKE_START = '(';

WebServer server(80);
AsyncWebServer asyncServer(81);
AsyncWebSocket controlWS("/control");

Servo throttleServo;
Servo steeringServo;
volatile int32_t ws_throttle_read = 1500;
volatile int32_t ws_steering_read = 1500;


volatile bool isClientConnected = false;
bool isFlashLightOn = false;
bool isRedLEDOn = false;


void setup()
{
  Serial.begin(115200);
  Serial.println();
  pinMode(RED_LED_PIN, OUTPUT);

  setupCamera();
  setupWifi();
  setupRoutes();
  setupServo();
  initWebSocket();

}


void onControlEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type,
             void *arg, uint8_t *data, size_t len) {
  switch (type) {
    case WS_EVT_CONNECT:
      Serial.printf("Control WebSocket client #%u connected from %s\n", client->id(), client->remoteIP().toString().c_str());
      break;
    case WS_EVT_DISCONNECT:
      Serial.printf("Control WebSocket client #%u disconnected\n", client->id());
      break;
    case WS_EVT_DATA:
      handleControlWebSocketMessage(arg, data, len);
      break;
    case WS_EVT_PONG:
    case WS_EVT_ERROR:
      break;
  }
}

void handleControlWebSocketMessage(void *arg, uint8_t *data, size_t len) {
  char buf[len] = "\0";
  memcpy(buf, data, len);
  
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
}



void initWebSocket() {  
  asyncServer.begin();
  controlWS.onEvent(onControlEvent);
  asyncServer.addHandler(&controlWS);
}


void loop()
{
  server.handleClient();
  controlWS.cleanupClients();
  writeToServo(); 
}

void writeToServo() {
   steeringServo.writeMicroseconds(ws_steering_read);    // tell servo to go to position 'steering'
   throttleServo.writeMicroseconds(ws_throttle_read);    // tell motor to drive with power = motorPower   
}

// setup functions

void setupServo() {
  ESP32PWM::timerCount[0]=4;
  ESP32PWM::timerCount[1]=4;
  throttleServo.setPeriodHertz(50);
  throttleServo.attach(THROTTLE_PIN, 1000, 2000);
  steeringServo.setPeriodHertz(50);    // standard 50 hz servo
  steeringServo.attach(STEERING_PIN, 1000, 2000); // attaches the servo on pin (whatever you assign)
}

void setupRoutes() {
  Serial.print("http://");
  Serial.println(WiFi.localIP());
  Serial.println("  /cam-lo.jpg");
  Serial.println("  /cam.mjpeg");
  Serial.println("  /cmd/<THROTTLE>,<STEERING>");

  server.on("/cam-lo.jpg", handleJpgLo);
  server.on("/cam.mjpeg", handleMjpeg);

  server.on(UriBraces("/cmd/{}"), handleCmd);
  server.begin();
}


void setupWifi() {
  WiFi.disconnect(true);

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting ...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    blinkRedLED();
    Serial.print(".");
  }
  digitalWrite(RED_LED_PIN, LOW);
  Serial.println("Connected!");
}
void setupCamera() {
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(loRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(10);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }
}

// handle routes
void handleCmd() {
  String argument = server.pathArg(0);
  
  char buf[argument.length()] = "\0";
  argument.toCharArray(buf, argument.length());
  char *token = strtok(buf, ",");
  if (token != NULL) {
    unsigned int curr_throttle_read = atoi(token+1);
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
  Serial.print(ws_throttle_read);
  Serial.print(" ");
  Serial.print(ws_steering_read);
  Serial.println();
  server.send(200, "text/plain","ack");
}

void handleJpgLo()
{
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

void handleMjpeg() {
  Serial.println("STREAM BEGIN");
  WiFiClient client = server.client();
  auto startTime = millis();
  int res = esp32cam::Camera.streamMjpeg(client);
  if (res <= 0) {
    Serial.printf("STREAM ERROR %d\n", res);
    return;
  }
  auto duration = millis() - startTime;
  Serial.printf("STREAM END %dfrm %0.2ffps\n", res, 1000.0 * res / duration);
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
  if (isRedLEDOn) {
    digitalWrite(RED_LED_PIN, HIGH);
    isRedLEDOn = false;
  } else {
    digitalWrite(RED_LED_PIN, LOW);
    isRedLEDOn = true;
  }
}
