// ======================= CẢM BIẾN — 1 = có gắn, 0 = không gắn =======================
// Gắn cảm biến nào thì đổi số 0 thành 1 và sửa chân (pin) cho đúng dây đã nối.
// Cảm biến cần thư viện: Arduino IDE -> Sketch -> Include Library -> Manage Libraries -> tìm đúng tên trong ngoặc.
#define USE_DHT       1   // DHT11/DHT22: nhiệt độ + độ ẩm       (thư viện "DHT sensor library" - Adafruit)
#define DHT_TYPE      DHT22 // đổi thành DHT11 nếu dùng DHT11 (màu xanh dương)
#define USE_DS18B20   0   // DS18B20: nhiệt độ (chống nước)         (thư viện "OneWire" + "DallasTemperature")
#define USE_BH1750    0   // BH1750: ánh sáng (lux) qua I2C         (thư viện "BH1750" - Christopher Laws)
#define USE_LDR       0   // quang trở: độ sáng thô 0-100 %
#define USE_PIR       0   // HC-SR501 / AM312: có người chuyển động
#define USE_GAS       0   // MQ-2 / MQ-135: khí gas / khói / chất lượng không khí (giá trị thô 0-100 %)
#define USE_FLAME     0   // cảm biến lửa (chân số, LOW = có lửa)
#define USE_SOUND     0   // cảm biến âm thanh (analog, 0-100 %)
#define USE_DOOR      0   // công tắc cửa từ (reed switch) — 1 = cửa mở
#define USE_SR04      0   // HC-SR04: khoảng cách (cm)
#define USE_RELAY     0   // relay: bật/tắt thiết bị từ dashboard
#define RELAY_ACTIVE_LOW 1 // đa số module relay kích mức LOW; nếu bật/tắt ngược thì đổi 0
#define USE_LED       0   // đèn LED / dải LED (qua MOSFET): chỉnh độ sáng 0-100 % từ dashboard
#define USE_BUZZER    0   // còi: dashboard bật/tắt (vd báo động)
