/* ARES — mạch cảm biến Wi-Fi (ESP32 / ESP8266: NodeMCU, Wemos D1...) — "{{NAME}}"
 * Mạch tự gửi số đo về máy chủ ARES qua Wi-Fi, nhận lệnh bật/tắt relay, đèn, còi từ dashboard.
 *
 * NẠP 1 LẦN DUY NHẤT (máy nào có Arduino IDE cũng được; nạp xong cắm mạch vào cục sạc bất kỳ):
 *  1. Arduino IDE -> File -> Preferences -> Additional boards manager URLs, thêm:
 *       ESP32  : https://espressif.github.io/arduino-esp32/package_esp32_index.json
 *       ESP8266: https://arduino.esp8266.com/stable/package_esp8266com_index.json
 *     Tools -> Board -> Boards Manager: cài "esp32" (Espressif) hoặc "esp8266".
 *  2. Sửa phần CẢM BIẾN bên dưới cho đúng mạch, cài thư viện cảm biến đã bật.
 *  3. Tools -> Board: chọn đúng loại (vd "ESP32 Dev Module", "NodeMCU 1.0"), Tools -> Port: cổng COM.
 *  4. Upload. Mở Serial Monitor (115200) để xem mạch kết nối. Dashboard -> Cảm biến sẽ hiện "Đang gửi".
 * Wi-Fi phải là 2.4 GHz (ESP không bắt được 5 GHz).
 */
#define WIFI_SSID     "{{WIFI_SSID}}"
#define WIFI_PASS     "{{WIFI_PASS}}"
#define ARES_SERVER   "{{SERVER}}"      // địa chỉ máy chủ ARES (http://IP:8200 trong mạng nội bộ, hoặc https://...ts.net)
#define DEVICE_TOKEN  "{{TOKEN}}"       // khoá riêng của mạch này — không chia sẻ
#define SEND_EVERY_MS 2000              // gửi số đo + nhận lệnh đèn mỗi 2 giây (đèn theo vùng phản hồi nhanh)

{{SENSORS}}

// ======================= CHÂN NỐI (sửa theo dây thật) =======================
#if defined(ESP32)
  #define DHT_PIN 4
  #define DS18B20_PIN 15
  #define LDR_PIN 34      // chân analog (ADC1: 32-39)
  #define PIR_PIN 27
  #define GAS_PIN 35
  #define FLAME_PIN 14
  #define SOUND_PIN 33
  #define DOOR_PIN 13
  #define TRIG_PIN 5
  #define ECHO_PIN 18
  #define RELAY_PIN 26
  #define LED_PIN 25
  #define BUZZER_PIN 23
  #define ADC_MAX 4095
#elif defined(ESP8266)
  // ESP8266 ít chân: D1/D2 dành cho I2C (BH1750). Chân ghi chung (vd LED_PIN = DOOR_PIN) thì chỉ bật 1 trong 2.
  #define DHT_PIN D4
  #define DS18B20_PIN D3
  #define LDR_PIN A0      // chỉ có 1 chân analog A0: chỉ bật 1 trong LDR / GAS / SOUND
  #define PIR_PIN D5
  #define GAS_PIN A0
  #define FLAME_PIN D6
  #define SOUND_PIN A0
  #define DOOR_PIN D7
  #define TRIG_PIN D0
  #define ECHO_PIN D8
  #define RELAY_PIN D6
  #define LED_PIN D7
  #define BUZZER_PIN D0
  #define ADC_MAX 1023
#else
  #error "File này cho ESP32/ESP8266. Arduino Uno/Nano/Mega: dùng code USB tải từ dashboard."
#endif

// ======================= KÊNH ĐÈN (dashboard tạo theo số đèn của mạch: mỗi kênh = 1 đèn) =======================
{{CHANNELS}}
#if N_CH > 0
int chVal[N_CH];  // 0..100: LED = % độ sáng, relay: > 0 là bật
#endif

#if defined(ESP32)
  #include <WiFi.h>
  #include <HTTPClient.h>
  #include <WiFiClientSecure.h>
#else
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
  #include <WiFiClientSecureBearSSL.h>
#endif
#if USE_DHT
  #include <DHT.h>
  DHT dht(DHT_PIN, DHT_TYPE);
#endif
#if USE_DS18B20
  #include <OneWire.h>
  #include <DallasTemperature.h>
  OneWire oneWire(DS18B20_PIN);
  DallasTemperature ds(&oneWire);
#endif
#if USE_BH1750
  #include <Wire.h>
  #include <BH1750.h>
  BH1750 lightMeter;
#endif

int relayOn = 0, ledPct = 0, buzzerOn = 0;
unsigned long lastSend = 0;

void applyOutputs() {
#if N_CH > 0
  for (int i = 0; i < N_CH; i++) {
    if (CH_PWM[i]) analogWrite(CH_PIN[i], map(constrain(chVal[i], 0, 100), 0, 100, 0, 255));
    else digitalWrite(CH_PIN[i], ((chVal[i] > 0) ^ RELAY_ACTIVE_LOW) ? HIGH : LOW);
  }
#endif
#if USE_RELAY
  digitalWrite(RELAY_PIN, (relayOn ^ RELAY_ACTIVE_LOW) ? HIGH : LOW);
#endif
#if USE_LED
  analogWrite(LED_PIN, map(ledPct, 0, 100, 0, 255));
#endif
#if USE_BUZZER
  digitalWrite(BUZZER_PIN, buzzerOn ? HIGH : LOW);
#endif
}

// tìm "key":số trong JSON trả về (không cần thư viện JSON)
int findInt(const String& s, const char* key, int def) {
  String k = String("\"") + key + "\":";
  int i = s.indexOf(k);
  if (i < 0) return def;
  return s.substring(i + k.length()).toInt();
}

void addNum(String& b, const char* key, float v, int digits) {
  if (isnan(v)) return;
  if (b.length() > 1) b += ",";
  b += "\""; b += key; b += "\":"; b += String(v, digits);
}

String readSensors() {
  String b = "{";
#if USE_DHT
  addNum(b, "temp", dht.readTemperature(), 1);
  addNum(b, "hum", dht.readHumidity(), 1);
#endif
#if USE_DS18B20
  ds.requestTemperatures();
  { float t = ds.getTempCByIndex(0); if (t > -100) addNum(b, USE_DHT ? "temp2" : "temp", t, 1); }
#endif
#if USE_BH1750
  { float lx = lightMeter.readLightLevel(); if (lx >= 0) addNum(b, "lux", lx, 0); }
#endif
#if USE_LDR
  addNum(b, "light", analogRead(LDR_PIN) * 100.0 / ADC_MAX, 0);
#endif
#if USE_PIR
  addNum(b, "motion", digitalRead(PIR_PIN) == HIGH ? 1 : 0, 0);
#endif
#if USE_GAS
  addNum(b, "gas", analogRead(GAS_PIN) * 100.0 / ADC_MAX, 0);
#endif
#if USE_FLAME
  addNum(b, "flame", digitalRead(FLAME_PIN) == LOW ? 1 : 0, 0);
#endif
#if USE_SOUND
  addNum(b, "sound", analogRead(SOUND_PIN) * 100.0 / ADC_MAX, 0);
#endif
#if USE_DOOR
  addNum(b, "door", digitalRead(DOOR_PIN) == HIGH ? 1 : 0, 0);
#endif
#if USE_SR04
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10); digitalWrite(TRIG_PIN, LOW);
  { unsigned long us = pulseIn(ECHO_PIN, HIGH, 30000); if (us > 0) addNum(b, "dist", us / 58.0, 0); }
#endif
#if USE_RELAY
  addNum(b, "relay", relayOn, 0);
#endif
#if USE_LED
  addNum(b, "led", ledPct, 0);
#endif
#if USE_BUZZER
  addNum(b, "buzzer", buzzerOn, 0);
#endif
#if N_CH > 0
  for (int i = 0; i < N_CH; i++) { char k[6]; sprintf(k, "ch%d", i + 1); addNum(b, k, chVal[i], 0); }
#endif
  addNum(b, "rssi", WiFi.RSSI(), 0);
  b += "}";
  return b;
}

bool sendToServer(const String& body) {
  String url = String(ARES_SERVER) + "/api/v1/sensor";
  HTTPClient http;
  WiFiClient plain;
#if defined(ESP32)
  WiFiClientSecure tls;
#else
  BearSSL::WiFiClientSecure tls;
#endif
  if (url.startsWith("https")) { tls.setInsecure(); http.begin(tls, url); }
  else { http.begin(plain, url); }
  http.setTimeout(8000);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Token", DEVICE_TOKEN);
  int code = http.POST(body);
  String resp = code > 0 ? http.getString() : "";
  http.end();
  Serial.printf("[ARES] gui %s -> HTTP %d\n", body.c_str(), code);
  if (code != 200) {
    if (code == 401) Serial.println("[ARES] Sai khoa thiet bi: tai lai code tu dashboard (mach co the da bi xoa).");
    return false;
  }
  int r = findInt(resp, "relay", relayOn), l = findInt(resp, "led", ledPct), z = findInt(resp, "buzzer", buzzerOn);
  bool chg = (r != relayOn || l != ledPct || z != buzzerOn);
  relayOn = r; ledPct = constrain(l, 0, 100); buzzerOn = z;
#if N_CH > 0
  for (int i = 0; i < N_CH; i++) { char k[6]; sprintf(k, "ch%d", i + 1); int v = constrain(findInt(resp, k, chVal[i]), 0, 100); if (v != chVal[i]) { chVal[i] = v; chg = true; } }
#endif
  if (chg) applyOutputs();
  return true;
}

void connectWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("[ARES] Ket noi Wi-Fi %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
  if (WiFi.status() == WL_CONNECTED) Serial.printf("\n[ARES] Da ket noi, IP %s\n", WiFi.localIP().toString().c_str());
  else Serial.println("\n[ARES] Chua ket noi duoc Wi-Fi (sai mat khau? Wi-Fi 5 GHz?) - se thu lai");
}

void setup() {
  Serial.begin(115200);
  delay(200);
#if USE_DHT
  dht.begin();
#endif
#if USE_DS18B20
  ds.begin();
#endif
#if USE_BH1750
  Wire.begin(); lightMeter.begin();
#endif
#if USE_PIR
  pinMode(PIR_PIN, INPUT);
#endif
#if USE_FLAME
  pinMode(FLAME_PIN, INPUT);
#endif
#if USE_DOOR
  pinMode(DOOR_PIN, INPUT_PULLUP);
#endif
#if USE_SR04
  pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);
#endif
#if USE_RELAY
  pinMode(RELAY_PIN, OUTPUT);
#endif
#if USE_LED
  pinMode(LED_PIN, OUTPUT);
#endif
#if USE_BUZZER
  pinMode(BUZZER_PIN, OUTPUT);
#endif
#if N_CH > 0
  for (int i = 0; i < N_CH; i++) { pinMode(CH_PIN[i], OUTPUT); chVal[i] = 0; }
#endif
  applyOutputs();
  connectWifi();
}

void loop() {
  if (millis() - lastSend >= SEND_EVERY_MS || lastSend == 0) {
    lastSend = millis();
    connectWifi();
    if (WiFi.status() == WL_CONNECTED) sendToServer(readSensors());
  }
  delay(20);
}
