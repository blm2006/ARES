/* ARES — mạch cảm biến cắm USB (Arduino Uno / Nano / Mega / Leonardo / Uno R4, cả ESP32/ESP8266 cắm USB) — "{{NAME}}"
 * Mạch gửi số đo qua cổng USB; dashboard ARES (Chrome/Edge hoặc app ARES Viewer) đọc cổng USB và gửi lên máy chủ.
 * Máy tính cắm mạch KHÔNG cần cài gì thêm — chỉ cần mở dashboard -> Cảm biến -> "Kết nối mạch qua USB".
 *
 * NẠP 1 LẦN DUY NHẤT bằng Arduino IDE:
 *  1. Sửa phần CẢM BIẾN bên dưới cho đúng mạch, cài thư viện cảm biến đã bật (Manage Libraries).
 *  2. Tools -> Board: chọn đúng loại (vd "Arduino Uno"), Tools -> Port: cổng COM của mạch. Upload.
 *  3. ĐÓNG Arduino IDE / Serial Monitor (2 chương trình không mở chung 1 cổng được), rồi kết nối trong dashboard.
 * Giao thức: mỗi 2 giây in 1 dòng JSON {"temp":28.5,...}; nhận dòng JSON {"relay":1,"led":50,"buzzer":0} để điều khiển.
 */
#define SEND_EVERY_MS 2000
#define BAUD 115200

{{SENSORS}}

// ======================= CHÂN NỐI (Arduino Uno/Nano — sửa theo dây thật) =======================
#define DHT_PIN 2
#define DS18B20_PIN 10
#define LDR_PIN A0
#define PIR_PIN 3
#define GAS_PIN A1
#define FLAME_PIN 4
#define SOUND_PIN A2
#define DOOR_PIN 8
#define TRIG_PIN 5
#define ECHO_PIN 6
#define RELAY_PIN 7
#define LED_PIN 9      // chân có dấu ~ (PWM): 3, 5, 6, 9, 10, 11
#define BUZZER_PIN 12
#if defined(ESP32)
  #define ADC_MAX 4095.0
#else
  #define ADC_MAX 1023.0
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
String line;
bool first = true;

void applyOutputs() {
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

int findInt(const String& s, const char* key, int def) {
  String k = String("\"") + key + "\":";
  int i = s.indexOf(k);
  if (i < 0) return def;
  return s.substring(i + k.length()).toInt();
}

void put(const char* key, float v, int digits) {
  if (isnan(v)) return;
  Serial.print(first ? "{\"" : ",\""); first = false;
  Serial.print(key); Serial.print("\":"); Serial.print(v, digits);
}

void sendLine() {
  first = true;
#if USE_DHT
  put("temp", dht.readTemperature(), 1);
  put("hum", dht.readHumidity(), 1);
#endif
#if USE_DS18B20
  ds.requestTemperatures();
  { float t = ds.getTempCByIndex(0); if (t > -100) put(USE_DHT ? "temp2" : "temp", t, 1); }
#endif
#if USE_BH1750
  { float lx = lightMeter.readLightLevel(); if (lx >= 0) put("lux", lx, 0); }
#endif
#if USE_LDR
  put("light", analogRead(LDR_PIN) * 100.0 / ADC_MAX, 0);
#endif
#if USE_PIR
  put("motion", digitalRead(PIR_PIN) == HIGH ? 1 : 0, 0);
#endif
#if USE_GAS
  put("gas", analogRead(GAS_PIN) * 100.0 / ADC_MAX, 0);
#endif
#if USE_FLAME
  put("flame", digitalRead(FLAME_PIN) == LOW ? 1 : 0, 0);
#endif
#if USE_SOUND
  put("sound", analogRead(SOUND_PIN) * 100.0 / ADC_MAX, 0);
#endif
#if USE_DOOR
  put("door", digitalRead(DOOR_PIN) == HIGH ? 1 : 0, 0);
#endif
#if USE_SR04
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10); digitalWrite(TRIG_PIN, LOW);
  { unsigned long us = pulseIn(ECHO_PIN, HIGH, 30000); if (us > 0) put("dist", us / 58.0, 0); }
#endif
#if USE_RELAY
  put("relay", relayOn, 0);
#endif
#if USE_LED
  put("led", ledPct, 0);
#endif
#if USE_BUZZER
  put("buzzer", buzzerOn, 0);
#endif
  put("up", millis() / 1000, 0);
  Serial.println("}");
}

void setup() {
  Serial.begin(BAUD);
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
  applyOutputs();
}

void loop() {
  while (Serial.available()) {  // lệnh từ dashboard: {"relay":1,"led":50}
    char c = Serial.read();
    if (c == '\n') {
      if (line.indexOf('{') >= 0) {
        relayOn = findInt(line, "relay", relayOn);
        ledPct = constrain(findInt(line, "led", ledPct), 0, 100);
        buzzerOn = findInt(line, "buzzer", buzzerOn);
        applyOutputs();
      }
      line = "";
    } else if (line.length() < 120) line += c;
  }
  if (millis() - lastSend >= SEND_EVERY_MS) { lastSend = millis(); sendLine(); }
}
