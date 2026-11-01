// ARES - Arduino đèn (bản LDR): đọc LDR ở A0, xuất PWM ở D9 theo lệnh từ máy tính (YOLO/ARES).
//
// Dây (giữ nguyên như bạn đã thử):  LDR + điện trở chia áp -> A0,  đèn qua MOSFET <- D9.
//
// Giao thức (115200 baud, mỗi dòng kết thúc bằng \n):
//   máy tính -> Arduino:  P,45.0              PWM mong muốn (%) do ARES tính từ số người + lux
//   Arduino -> máy tính:  {"lux_zones":[412],"lux_window":412}    mỗi 0,5 giây
//
// Khác bản chạy riêng: Arduino KHÔNG tự quyết độ sáng nữa. ARES nhìn camera (có người hay không) và đọc lux từ đây,
// rồi tự chỉnh PWM để lux mặt bàn đạt mục tiêu. Trời càng sáng thì PWM càng giảm (tiết kiệm điện).
//
// Fail-safe: quá FAILSAFE_MS không có lệnh (máy tính treo, rút cáp) -> đèn về FALLBACK_PCT.

const int SENSOR_PIN = A0;
const int LED_PIN = 9;

// ---- hiệu chuẩn LDR -> lux (quan trọng: chỉnh theo phòng của bạn) ------------------------------
const int RAW_DARK = 640;          // SAU KHI ĐẢO: tay che kín cảm biến (~380 thô) -> 0 lux
const int RAW_BRIGHT = 820;        // SAU KHI ĐẢO: phòng sáng bình thường (~200 thô trở xuống) -> LUX_AT_BRIGHT
const float LUX_AT_BRIGHT = 400.0; // lux thật tại RAW_BRIGHT: đo bằng app đo lux trên điện thoại đặt cạnh LDR
// LDR không tuyến tính nên đây chỉ là xấp xỉ 2 điểm. Đủ dùng cho điều khiển đèn, không phải đồng hồ đo lux.

const unsigned long FAILSAFE_MS = 45000;
const float FALLBACK_PCT = 70.0;
const unsigned long SENSOR_PERIOD_MS = 500;
const int STEP_PWM = 2;            // sáng dần cho dịu mắt: 2/255 mỗi 30 ms (~3,8 s từ tắt lên 100 %). Tăng số này nếu muốn sáng nhanh hơn
const unsigned long STEP_MS = 30;

int currentPWM = 0;
int targetPWM = 0;
unsigned long lastCmd = 0, lastSensor = 0, lastStep = 0;
char buf[32];
uint8_t blen = 0;

void applyLine(char *line) {
  if (line[0] != 'P' || line[1] != ',') return;
  float pct = constrain(atof(line + 2), 0.0f, 100.0f);
  targetPWM = (int)(pct * 2.55f + 0.5f);
  lastCmd = millis();
}

const bool INVERT = true;          // đặt true nếu module của bạn cho số NHỎ khi sáng, số LỚN khi tối
float lastRaw = 0;

float readLux() {
  long total = 0;
  for (int i = 0; i < 10; i++) {
    total += analogRead(SENSOR_PIN);
    delay(1);
  }
  float raw = total / 10.0f;
  if (INVERT) raw = 1023.0f - raw;
  lastRaw = raw;
  float lux = (raw - RAW_DARK) * LUX_AT_BRIGHT / (float)(RAW_BRIGHT - RAW_DARK);
  return lux < 0 ? 0 : lux;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  analogWrite(LED_PIN, 0);
  lastCmd = millis();
  Serial.println(F("# ares_led (LDR) ready"));
}

void loop() {
  unsigned long now = millis();

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { buf[blen] = 0; applyLine(buf); blen = 0; }
    else if (c != '\r' && blen < sizeof(buf) - 1) buf[blen++] = c;
    else if (blen >= sizeof(buf) - 1) blen = 0;
  }

  int goal = ((now - lastCmd) > FAILSAFE_MS) ? (int)(FALLBACK_PCT * 2.55f) : targetPWM;

  if (now - lastStep >= STEP_MS) {
    lastStep = now;
    if (currentPWM < goal) currentPWM = min(currentPWM + STEP_PWM, goal);
    else if (currentPWM > goal) currentPWM = max(currentPWM - STEP_PWM, goal);
    analogWrite(LED_PIN, currentPWM);
  }

  if (now - lastSensor >= SENSOR_PERIOD_MS) {
    lastSensor = now;
    long l = (long)(readLux() + 0.5f);
    Serial.print(F("{\"lux_zones\":["));
    Serial.print(l);
    Serial.print(F("],\"lux_window\":"));
    Serial.print(l);
    Serial.print(F(",\"raw\":"));
    Serial.print((int)lastRaw);   // số đọc thô A0 (0..1023), dùng để hiệu chuẩn RAW_DARK / RAW_BRIGHT
    Serial.println('}');
  }
}
