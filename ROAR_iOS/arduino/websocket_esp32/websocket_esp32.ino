#include <WiFi.h>
#include <WebServer.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <esp32cam.h>

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

#include <ESP32Servo.h>



const char HANDSHAKE_START = '(';
char* WIFI_SSID = "NETGEAR05";
char* WIFI_PASS = "wuxiaohua1011";


WebServer server(80);
AsyncWebServer wsServer(81);
AsyncWebSocket ws("/ws");

static auto loRes = esp32cam::Resolution::find(320, 240);
static auto hiRes = esp32cam::Resolution::find(800, 600);


volatile int32_t ws_throttle_read = 1500;
volatile int32_t ws_steering_read = 1500;


//-------------Motor control-------------------
Servo motorControl;  // create servo object to control a servo
// 16 servo objects can be created on the ESP32
String input_motor; // this gets the input from serial
int motorPower = 90;    // 90 = stopped, 0 = full reverse, 180 = full forward
// Recommended PWM GPIO pins on the ESP32 include 2,4,12-19,21-23,25-27,32-33 
// Don't use 4 - connected to LED
int motorOutputPin = 12;
int MAXMOTOR = 2000; //Should be as high as 2000 
int MINMOTOR = 1000; //Should be as low as 1000

//--------------setup servo for steering---------------
Servo steeringServo;  // create servo object to control a servo
// 16 servo objects can be created on the ESP32
String input_steering; // this gets the input from serial
int steering = 90;    // Steering servo initialize going straight = 90 degrees
// Recommended PWM GPIO pins on the ESP32 include 2,4,12-19,21-23,25-27,32-33 
// Don't use 4 - connected to LED
int SteeringOutputPin = 13;
int MAX = 2000; //Should be as high as 2000 but my steering is a little broken
int MIN = 1000; //Should be as low as 1000

const int LED = 4; // Built in LED


void setup() {
  esp_bt_controller_deinit();
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(LED, OUTPUT); // Used for trouble-shooting, etc...
  startCamera();
  startWebserver();
  startServo();
}

void loop() {
  // put your main code here, to run repeatedly:
  server.handleClient();  
  ws.cleanupClients();
  writeToServo();
}

void startServo() {
//  ESP32PWM::allocateTimer(0);
//  ESP32PWM::allocateTimer(1);
//  ESP32PWM::allocateTimer(2);
//  ESP32PWM::allocateTimer(3);
  motorControl.setPeriodHertz(50);
  motorControl.attach(motorOutputPin, MINMOTOR, MAXMOTOR);
  steeringServo.setPeriodHertz(50);    // standard 50 hz servo
  steeringServo.attach(SteeringOutputPin, MIN, MAX); // attaches the servo on pin (whatever you assign)
}

void writeToServo() {
   steeringServo.writeMicroseconds(ws_steering_read);    // tell servo to go to position 'steering'
   motorControl.writeMicroseconds(ws_throttle_read);    // tell motor to drive with power = motorPower   
}


void handleWebSocketMessage(void *arg, uint8_t *data, size_t len) {
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
  Serial.print(ws_throttle_read);
  Serial.print(" ");
  Serial.print(ws_steering_read);
  Serial.println();
}
void onEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type,
             void *arg, uint8_t *data, size_t len) {
  switch (type) {
    case WS_EVT_CONNECT:
      Serial.printf("WebSocket client #%u connected from %s\n", client->id(), client->remoteIP().toString().c_str());
      break;
    case WS_EVT_DISCONNECT:
      Serial.printf("WebSocket client #%u disconnected\n", client->id());
      break;
    case WS_EVT_DATA:
      handleWebSocketMessage(arg, data, len);
      break;
    case WS_EVT_PONG:
    case WS_EVT_ERROR:
      break;
  }
}

void startCamera() {
    {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(80);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }
}


void startWebserver(){
  // delete old config
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Attempting to connect ..");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("Success!");
  Serial.print("IP Addr: ");
  Serial.println(WiFi.localIP());
  
  Serial.print("Available routes:");
  Serial.println("  /cam.bmp");
  Serial.println("  /cam-lo.jpg");
  Serial.println("  /cam-hi.jpg");
  Serial.println("  /cam.mjpeg");

  server.on("/cam.bmp", handleBmp);
  server.on("/cam-lo.jpg", handleJpgLo);
  server.on("/cam-hi.jpg", handleJpgHi);
  server.on("/cam.jpg", handleJpg);
  server.on("/cam.mjpeg", handleMjpeg);

  ws.onEvent(onEvent);
  wsServer.addHandler(&ws);
  wsServer.begin();
  server.begin();
}


void handleBmp()
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
  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));

  if (!frame->toBmp()) {
    Serial.println("CONVERT FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CONVERT OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));

  server.setContentLength(frame->size());
  server.send(200, "image/bmp");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

void serveJpg()
{
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
//  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
//                static_cast<int>(frame->size()));

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

void handleJpgLo()
{
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  serveJpg();
}

void handleJpgHi()
{
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}

void handleJpg()
{
  server.sendHeader("Location", "/cam-hi.jpg");
  server.send(302, "", "");
}

void handleMjpeg()
{
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }

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
