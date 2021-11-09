#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include <Wire.h>

#include <WiFi.h>
#include <WebServer.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <esp32cam.h>


#define I2C_SDA 14
#define I2C_SCL 2

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

char* WIFI_SSID = "NETGEAR78";
char* WIFI_PASS = "wuxiaohua1011";


Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Adafruit_MPU6050 mpu;
AsyncWebServer wsServer(81);
AsyncWebSocket websocket("/control");

volatile float ax = 0;
volatile float ay = 0;
volatile float az = 0;

void setup(void) {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);
  setUpMPU6050();
  setupDisplay();
  startWebserver();

  
}

void loop() {
  /* Get new sensor events with the readings */
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  ax = a.acceleration.x;
  ay = a.acceleration.y;
  az = a.acceleration.z;

  /* Print out the values */
  String content = String("Acceleration: \n") + 
                   getAccelerometerReading(a) + String("\n") + 
                   String("IP: ") + String(IpAddress2String(WiFi.localIP()));
  Serial.println(content);
  printToDisplay(content);

}


String getAccelerometerReading(sensors_event_t a) {
   return String(a.acceleration.x) + String(", ") + String(a.acceleration.y) + String(", ") + String(a.acceleration.z);
}

void notFound(AsyncWebServerRequest *request) {
    request->send(404, "text/plain", "Not found");
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


  wsServer.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
        request->send(200, "text/plain",  String(ax) + String(",") + String(ay) + String(",") + String(az));
    });

  wsServer.onNotFound(notFound);
  wsServer.begin();
}



void setupDisplay() {
  while(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    delay(100);
  }
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  printToDisplay("Board Starting...");
}


void printToDisplay(String content) {
  display.clearDisplay();
  display.setCursor(0,0);
  display.println(content);
  display.display();
}

void setUpMPU6050() {
  Serial.println("Adafruit MPU6050 test!");
  // Try to initialize!
  while (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    delay(100);
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");
  delay(100);
}

String IpAddress2String(const IPAddress& ipAddress)
{
  return String(ipAddress[0]) + String(".") +\
  String(ipAddress[1]) + String(".") +\
  String(ipAddress[2]) + String(".") +\
  String(ipAddress[3])  ; 
}
