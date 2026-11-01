"""Ghép mọi thành phần ARES lại và chạy bằng các luồng độc lập.

    vision      full FPS (vision_fps), 1 FPS khi đang train   Khung hình -> YOLO -> vùng -> 2 bộ đệm   (vision_tracker, fps)
    lighting    mỗi lighting_interval_s Vòng PWM tỉ lệ -> MQTT                               (control_engine)
    hvac        mỗi hvac_tick_s         Pre-cooling / setpoint / fail-safe -> MQTT (IR)      (control_engine)
    telemetry   mỗi telemetry_interval_s Ghi telemetry_logs
    nightly     mỗi 30 giây             Gom thói quen + retrain khi đến giờ
    housekeep   mỗi 5 giây              Làm mới số liệu AL, đồng hồ lịch
    sim_world   (chỉ SimulationMode)    Quay vật lý giả
    + luồng ghi DB, luồng lưu ảnh AL, luồng MQTT (paho), luồng dashboard (uvicorn)

Quy tắc: không luồng nào gọi hàm chặn của luồng khác. YOLO chậm hay train ban đêm thì đèn và điều hòa
vẫn chạy (chúng chỉ đọc bộ đệm + cảm biến mới nhất qua SharedState có khóa).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from ares.active_learner import ActiveLearner
from ares.buffers import OccupancyBuffer
from ares.cameras import MAIN_ID, CameraHub, CameraSpec
from ares.clock import Clock, RealClock, SimClock
from ares.config import AresSettings, load_room
from ares.control_engine import ControlEngine
from ares.db_manager import DBManager
from ares.fps import FpsGovernor
from ares.model_registry import ModelRegistry
from ares.nightly import NightlyJobs
from ares.pca_filter import PcaFilter
from ares.retrain import Retrainer
from ares.schedule import StaticSchedule
from ares.state import Event, SensorReading, SharedState
from ares.vision_tracker import VisionTracker
from ares.zones import ZoneMap

log = logging.getLogger(__name__)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def make_clock(cfg: AresSettings) -> Clock:
    if cfg.simulation_mode:
        start = None
        if cfg.sim_start:
            start = datetime.fromisoformat(cfg.sim_start)
        return SimClock(cfg.tz, start=start, speed=cfg.sim_speed)
    return RealClock(cfg.tz)


class AresRuntime:
    def __init__(self, cfg: AresSettings, *, clock: Clock | None = None, zonemap: ZoneMap | None = None,
                 schedule=None, db: DBManager | None = None, bus=None, source=None, detector=None,
                 detector_factory: Callable | None = None, world=None, scenario=None,
                 registry: ModelRegistry | None = None, with_dashboard: bool = True):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.sim = bool(cfg.simulation_mode)
        self.clock = clock or make_clock(cfg)
        self.zonemap = zonemap or ZoneMap(load_room(cfg.zones_file, cfg.room_id))
        self.n = self.zonemap.n
        self.schedule = schedule or StaticSchedule.from_file(cfg.schedule_file, cfg.tz)
        self.state = SharedState(self.n, cfg.room_id, self.sim)
        self.db = db or DBManager(cfg.database_url, cfg.room_id, self.clock, cfg.habit_slot_min, cfg.habit_weeks,
                                  cfg.habit_min_people)
        self.light_buf = OccupancyBuffer(cfg.lighting_window_s, self.n, cfg.buffer_max_gap_s)
        self.hvac_buf = OccupancyBuffer(cfg.hvac_window_s, self.n, cfg.buffer_max_gap_s)
        self.registry = registry or ModelRegistry(cfg.models_dir, cfg.weights_path())
        self.world, self.scenario = world, scenario
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._dash_server = None
        self._ended_warned = False
        self.started_at: float | None = None
        self._retrain_thread: threading.Thread | None = None

        # ---- nguồn vào / bus ---------------------------------------------------------------
        if self.sim:
            bus, source, detector = self._build_sim(bus, source, detector)
            detector_factory = None
        else:
            if bus is None:
                if cfg.bus == "serial":
                    from ares.serial_bus import SerialBus
                    bus = SerialBus(cfg, self.clock, self.n, on_sensor=self._on_sensor, on_connection=self._on_connection)
                else:
                    from ares.mqtt_client import MqttBus
                    bus = MqttBus(cfg, self.clock, self.n, on_sensor=self._on_sensor, on_connection=self._on_connection)
            if detector is None:
                from ares.detector import YoloDetector
                ver, path = self.registry.production()
                detector = YoloDetector(cfg, path, ver)
                detector_factory = detector_factory or (lambda p, v: YoloDetector(cfg, p, v))
            if source is None:
                from ares.vision_tracker import CvFrameSource
                source = CvFrameSource("ares", cfg.camera_source, cfg.camera_loop, 1.0 / cfg.vision_fps)
        self.bus = bus

        # ---- động cơ điều khiển + các dịch vụ -----------------------------------------------
        self.engine = ControlEngine(cfg, self.clock, self.state, self.zonemap, self.light_buf, self.hvac_buf,
                                    self.schedule, self.db, self.bus, self.db)
        emit = self.engine.emit
        from ares.tuning import Tuning
        self.tuning = Tuning(cfg, emit=emit, clock=self.clock)
        self.tuning.load()  # giá trị delay / vùng điều chỉnh đã chỉnh trên dashboard (đè lên ares.env)
        self.pca = PcaFilter(cfg, self.clock, self.db, emit=emit) if cfg.pca_enabled else None
        self.al = ActiveLearner(cfg, self.clock, self.db, emit=emit, pca=self.pca)
        self.tracker = VisionTracker(
            cfg, self.clock, self.zonemap, source, detector, self.light_buf, self.hvac_buf, self.state,
            al=self.al, registry=None if self.sim else self.registry, detector_factory=detector_factory,
            emit=emit, compute_brightness=True, pca=self.pca, cam_id=MAIN_ID, async_model_load=not self.sim)
        self.retrainer = Retrainer(cfg, self.clock, self.al, self.registry, emit=emit, pca=self.pca)
        self.fps = FpsGovernor(cfg, lambda: self.retrainer.running, clock=self.clock, emit=emit)
        self.tracker.period_fn = self.fps.vision_period
        self.nightly = NightlyJobs(cfg, self.clock, self.db, self.al, self.retrainer, emit=emit, pca=self.pca)
        self.with_dashboard = with_dashboard

        # ---- nhiều camera (dashboard kiểu CCTV) -----------------------------------------------
        self._sim_cams: dict[str, tuple] = {}
        hub_kwargs: dict = {}
        if self.sim:
            hub_kwargs = {"source_factory": lambda sp: self._sim_cams[sp.id][1],
                          "detector_for": lambda sp: self._sim_cams[sp.id][2]}
        else:
            hub_kwargs = {"registry": self.registry, "detector_factory": detector_factory}
        self.hub = CameraHub(cfg, self.clock, self.state, al=self.al, pca=self.pca, emit=emit, **hub_kwargs)
        self.hub.period_fn = self.fps.hub_period
        if self.sim:
            self._add_sim_cameras()

    # ================================================================== dựng phần giả
    def _build_sim(self, bus, source, detector):
        from ares.simulation import OccupancyScenario, SimBus, SimDetector, SimFrameSource, SimWorld

        cfg = self.cfg
        self.scenario = self.scenario or OccupancyScenario(cfg, self.zonemap, self.schedule, seed=cfg.sim_seed)
        self.world = self.world or SimWorld(cfg, self.clock, self.zonemap, self.scenario, seed=cfg.sim_seed)
        self.world.on_sensor = self._on_sensor
        return (bus or SimBus(self.world),
                source or SimFrameSource(cfg, self.clock, self.scenario, self.world),
                detector or SimDetector(cfg, self.clock, self.scenario, seed=cfg.sim_seed + 1))

    SIM_CAM_NAMES = ("Phòng B", "Hành lang", "Sảnh chính", "Thư viện", "Phòng họp", "Căng tin", "Cổng trường", "Bãi xe",
                     "Phòng máy", "Xưởng thực hành", "Giảng đường A", "Giảng đường B", "Khu tự học", "Lối thoát hiểm", "Sân sau")

    def _add_sim_cameras(self) -> None:
        """Camera giả thứ 2..N: mỗi camera một kịch bản người riêng (cùng lịch lớp), để thử lưới CCTV."""
        from ares.simulation import OccupancyScenario, SimDetector, SimFrameSource

        cfg = self.cfg
        for k in range(2, cfg.sim_cameras + 1):
            sc = OccupancyScenario(cfg, self.zonemap, self.schedule, seed=cfg.sim_seed + 97 * k,
                                   attendance=(0.10 + 0.04 * (k % 6), 0.45 + 0.05 * (k % 8)))
            name = self.SIM_CAM_NAMES[(k - 2) % len(self.SIM_CAM_NAMES)]
            if k - 2 >= len(self.SIM_CAM_NAMES):
                name = f"{name} {k}"
            spec = CameraSpec(f"sim{k}", name, "sim", "sim")
            self._sim_cams[spec.id] = (sc, SimFrameSource(cfg, self.clock, sc, None),
                                       SimDetector(cfg, self.clock, sc, seed=cfg.sim_seed + 1 + 97 * k))
            self.hub.add(spec)

    # ================================================================== callback
    def _on_sensor(self, r: SensorReading) -> None:
        self.state.set_sensors(r)

    def _on_connection(self, ok: bool) -> None:
        self.state.mqtt_connected = ok
        self.engine.emit(Event(self.clock.time(), "MQTT_UP" if ok else "MQTT_DOWN", "info" if ok else "warn",
                               "Đã kết nối MQTT broker." if ok else "Mất kết nối MQTT broker — đèn/điều hòa giữ trạng thái cuối."))

    # ================================================================== vòng đời
    def _check_dashboard_exposure(self) -> None:
        cfg = self.cfg
        if cfg.dashboard_host not in LOCAL_HOSTS and not cfg.dashboard_token:
            raise RuntimeError(
                f"Dashboard sẽ mở ra mạng ({cfg.dashboard_host}) mà chưa đặt ARES_DASHBOARD_TOKEN. "
                "Dashboard hiển thị hình ảnh người trong phòng và cho phép duyệt ảnh/đổi model, nên bắt buộc có token. "
                "Đặt ARES_DASHBOARD_TOKEN=<chuỗi bí mật> hoặc dùng ARES_DASHBOARD_HOST=127.0.0.1."
            )

    def start(self) -> None:
        cfg = self.cfg
        if self.with_dashboard:
            self._check_dashboard_exposure()
        self.started_at = self.clock.time()
        self.al.start()
        self.bus.start()
        if not self.sim:
            self.hub.rescan()  # dò camera USB cắm thêm + đọc cameras.json (nền, không chặn khởi động)
        mode = "MÔ PHỎNG (không cần ESP32/camera)" if self.sim else "THẬT"
        self.engine.emit(Event(self.clock.time(), "SYSTEM_START", "info",
                               f"ARES khởi động, phòng {cfg.room_id}, chế độ {mode}, model {self.state.model_version}.",
                               {"simulation": self.sim, "room": cfg.room_id}))
        spec: list[tuple[str, float | Callable[[], float], Callable[[], None]]] = [
            ("vision", self.fps.vision_period, self._vision_tick),
            ("lighting", lambda: cfg.lighting_interval_s, self.engine.lighting_tick),  # chỉnh được khi đang chạy
            ("hvac", cfg.hvac_tick_s, self.engine.hvac_tick),
            ("telemetry", cfg.telemetry_interval_s, self.log_telemetry),
            ("hub", self.fps.hub_period, self.hub.tick),
            ("housekeep", 5.0, self._housekeep),
            ("nightly", 30.0, self.nightly.tick),
        ]
        if self.sim:
            spec.insert(0, ("sim_world", 5.0, self._sim_tick))
        for name, period, fn in spec:
            th = threading.Thread(target=self._worker, args=(name, period, fn), name=f"ares-{name}", daemon=True)
            th.start()
            self._threads.append(th)
        if self.with_dashboard:
            self._start_dashboard()

    def _worker(self, name: str, period_s: float | Callable[[], float], fn: Callable[[], object]) -> None:
        clock = self.clock
        next_t = clock.time()
        while not self._stop.is_set():
            period = period_s() if callable(period_s) else period_s
            try:
                fn()
                self.state.beat(name, clock.time())
            except Exception as e:  # noqa: BLE001 - một vòng lỗi không được giết cả luồng
                log.exception("Luồng %s lỗi (bỏ vòng này)", name)
                self.state.beat(name, clock.time(), error=f"{type(e).__name__}: {e}")
            next_t += period
            delay = next_t - clock.time()
            if delay < 0:  # chạy chậm hơn chu kỳ: không dồn bù
                next_t, delay = clock.time(), 0.0
            self._stop.wait(clock.real_timeout(delay))

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=8)
        if self._dash_server is not None:
            self._dash_server.should_exit = True
        try:
            self.engine.emit(Event(self.clock.time(), "SYSTEM_STOP", "info", "ARES dừng."))
        except Exception:
            pass
        for fn in (self.tracker.close, self.hub.close, lambda: self.al.drain(5), self.al.stop,
                   (self.pca.close if self.pca else (lambda: None)), self.bus.stop):
            try:
                fn()
            except Exception:
                log.exception("Lỗi khi dừng")
        self.db.close()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("Nhận Ctrl+C — đang dừng…")
        finally:
            self.stop()

    # ================================================================== các vòng
    def _sim_tick(self) -> None:
        self.world.advance(self.clock.time())

    def _vision_tick(self) -> None:
        src = self.tracker.source
        if getattr(src, "exhausted", False):
            if not self._ended_warned:
                self._ended_warned = True
                self.engine.emit(Event(self.clock.time(), "VIDEO_ENDED", "warn",
                                       "Nguồn video đã phát hết (camera_loop=false) — vision dừng nhận khung hình mới."))
            return
        self.tracker.tick()

    def _housekeep(self) -> None:
        self.state.al_stats = self.al.stats()
        self.state.extra["fps"] = self.fps.status()
        if self.pca is not None:
            self.pca.tick()
            self.state.extra["pca"] = self.pca.status()
        else:
            self.state.extra["pca"] = {"enabled": False}
        if hasattr(self.schedule, "maybe_reload"):
            self.schedule.maybe_reload()
        if self.bus is not None:
            self.state.mqtt_connected = bool(getattr(self.bus, "connected", False))

    def log_telemetry(self) -> None:
        """Một dòng telemetry_logs (spec mục 5). Mất camera -> occ_total = NULL, không ghi 0."""
        t = self.clock.time()
        cfg = self.cfg
        stats = self.light_buf.stats(t)
        vision_ok = stats.ok and stats.last_t is not None and (t - stats.last_t) <= cfg.frame_stale_s
        sensors = self.state.get_sensors()
        fresh = sensors is not None and (t - sensors.t) <= cfg.sensor_stale_s
        lighting = self.state.get_lighting()
        ac = self.state.get_ac()
        row = {
            "timestamp": self.clock.naive_now().replace(microsecond=0),
            "occ_total": int(round(stats.mean_total)) if vision_ok else None,
            "occ_zones": [int(round(x)) for x in stats.mean_zones] if vision_ok else [],
            "temp_in": sensors.temp_in if fresh else None,
            "temp_out": sensors.temp_out if fresh else None,
            "humidity": sensors.humidity if fresh else None,
            "lux_window": sensors.lux_window if fresh else None,
            "lux_zones": [None if v is None else round(v, 1) for v in sensors.lux_zones] if fresh else [],
            "ac_state": ac.as_json() if ac.known else {"power": 0, "temp": 0, "fan": 0},
            "led_pwm": list(lighting.pwm),
        }
        self.db.log_telemetry(row)

    # ================================================================== dashboard
    def _start_dashboard(self) -> None:
        import uvicorn

        from ares.dashboard.app import create_app

        cfg = self.cfg
        app = create_app(self)
        config = uvicorn.Config(app, host=cfg.dashboard_host, port=cfg.dashboard_port, log_level="warning",
                                access_log=False)
        self._dash_server = uvicorn.Server(config)
        th = threading.Thread(target=self._dash_server.run, name="ares-dashboard", daemon=True)
        th.start()
        self._threads.append(th)
        log.info("Dashboard: http://%s:%s", "localhost" if cfg.dashboard_host in ("0.0.0.0", "::") else cfg.dashboard_host,
                 cfg.dashboard_port)

    # ================================================================== truy vấn cho dashboard
    def retrain_async(self, force: bool = True) -> dict:
        if self.retrainer.running:
            return {"started": False, "reason": "Một lần train khác đang chạy."}
        th = threading.Thread(target=lambda: self.retrainer.run("manual", force=force), name="ares-retrain-manual",
                              daemon=True)
        th.start()
        self._retrain_thread = th
        return {"started": True}

    def status(self) -> dict:
        now = self.clock.now()
        t = self.clock.time()
        snap = self.state.snapshot()
        cur = self.schedule.current(now)
        nxt = self.schedule.next_within(now, 24 * 60)
        snap.update({
            "now": now.isoformat(timespec="seconds"),
            "t": t,
            "speed": getattr(self.clock, "speed", 1.0),
            "uptime_s": None if self.started_at is None else round(t - self.started_at),
            "schedule": {
                "today": self.schedule.today(now) if hasattr(self.schedule, "today") else [],
                "current": None if cur is None else {"name": cur.name, "start": cur.start.isoformat(), "end": cur.end.isoformat()},
                "next": None if nxt is None else {"name": nxt.name, "start": nxt.start.isoformat(), "end": nxt.end.isoformat()},
            },
            "zones": [{"id": z.id, "name": z.name, "capacity": z.capacity, "k": z.k} for z in self.zonemap.room.zones],
            "n_max": self.zonemap.room.n_max,
            "limits": {
                "lux_occupied": self.cfg.lux_occupied, "lux_dense": self.cfg.lux_dense, "lux_idle": self.cfg.lux_idle,
                "t_min": self.cfg.t_min, "t_max": self.cfg.t_max, "deadband": self.cfg.deadband,
            },
            "retrain": {"running": self.retrainer.running, "last": self.retrainer.last_result,
                        "time": self.cfg.retrain_time, "nightly": self.nightly.last_summary},
            "mqtt_devices": dict(getattr(self.bus, "device_status", {}) or {}),
            "db": {"dropped": self.db.dropped, "write_errors": self.db.write_errors},
            "events": self.state.recent_events(30),
        })
        if self.sim and self.world is not None:
            m = self.world.metrics
            snap["sim"] = {
                "led_kwh": round(m.led_wh / 1000, 3), "ac_kwh": round(m.ac_wh / 1000, 3),
                "occupied_min": round(m.occupied_min), "hot_min": round(m.hot_min), "warm_min": round(m.warm_min),
                "dark_zone_min": round(m.dark_zone_min), "ac_on_min": round(m.ac_on_min),
                "true_people": len(self.scenario.people_at(now)),
                "true_t_in": round(self.world.t_in, 2) if self.world.t_in is not None else None, "true_pwm": [round(p, 1) for p in self.world.pwm],
                "true_desk_lux": [round(x) for x in self.world.last_desk],
            }
        return snap
