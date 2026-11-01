"""ARES — Adaptive Real-time Energy-saving System.

Hệ thống quản lý vi khí hậu và năng lượng theo người dùng (Occupant-Centric BEMS):
YOLO đếm người theo vùng -> bộ điều khiển đèn (daylight harvesting) và điều hòa
(pre-cooling + setpoint động) -> MQTT -> ESP32. Học chủ động (human-in-the-loop)
để YOLO tốt dần theo phòng thật.

Nằm cạnh hệ thống yolo-al cũ, không sửa code cũ.
"""

__version__ = "0.1.0"
