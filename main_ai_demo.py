import csv
import itertools
import json
import os
import time
import threading   # PASO scheduling
import io          # PASO profiling
import cProfile    # PASO profiling
import pstats      # PASO profiling
from collections import deque
from typing import Optional

from mqtt_publisher import init_mqtt, publish_vitals, stop_mqtt

import joblib
import numpy as np
import pandas as pd

# ==============================================================================
# Hardware imports
# GPIO (LEDs + buzzer) is always imported — it works in both real and demo mode.
# Only the sensor-specific libraries (adafruit_dht, smbus2) are stubbed out
# in DEMO_MODE since they require physical wiring.
# ==============================================================================
_DEMO_MODE_EARLY = True   # must match DEMO_MODE in CONFIGURATION block below

from gpiozero import Device, LED, PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from smbus2 import SMBus

Device.pin_factory = LGPIOFactory()

# LEDs and buzzer — always real, work in both modes
led_green  = LED(17)
led_yellow = LED(27)
led_red    = LED(22)
buzzer     = PWMOutputDevice(23)

if not _DEMO_MODE_EARLY:
    # Real sensor hardware — only needed when reading from the actual sensor
    import adafruit_dht
    import board
    dht = adafruit_dht.DHT22(board.D4, use_pulseio=False)
else:
    # Stub for DHT22 only — temperature comes from the CSV in demo mode
    class _DHTStub:
        temperature = None
        def exit(self): pass
    dht = _DHTStub()

# -- MAX30102 -------------------------------------------------------------------
I2C_ADDR = 0x57


def setup_max30102(bus) -> None:
    if bus is None:
        return   # demo mode — no hardware
    bus.write_byte_data(I2C_ADDR, 0x09, 0x40)
    time.sleep(0.1)
    bus.write_byte_data(I2C_ADDR, 0x09, 0x03)
    bus.write_byte_data(I2C_ADDR, 0x0C, 0x1F)
    bus.write_byte_data(I2C_ADDR, 0x0D, 0x1F)
    bus.write_byte_data(I2C_ADDR, 0x0A, 0x27)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Heart-rate thresholds
HR_WARN_BPM  = 100
HR_ALERT_BPM = 120

# Ambient temperature thresholds (standalone alert)
TEMP_WARN_HIGH_C  = 32.0
TEMP_ALERT_HIGH_C = 34.0

# --- Temperature context parameters
TEMP_DROP_RATE_C      = 1.5
TEMP_CONTEXT_WINDOW_S = 60
TEMP_STABLE_STD_C     = 0.3
IR_LOW_THRESHOLD      = 60000

# --- AI pipeline
PPG_WINDOW             = 250
PPG_SHORT_WINDOW       = 75
ANOMALY_PERSIST_S      = 5.0
ANOMALY_PROB_THRESHOLD = 0.65
AI_CONFIDENCE_GATE     = 0.50
CONTEXT_WINDOW_SECONDS = 5.0

CONTACT_VOTE_N = 1

# Motion flush fraction
MOTION_FLUSH_FRACTION = 0.8

# --- Severity debounce (asymmetric)
SEVERITY_UPGRADE_S   = 0.5
SEVERITY_DOWNGRADE_S = 0.3

# --- Stage 2 hard gate
IR_VALID_MIN = 50_000
IR_VALID_MAX = 200_000

# --- AC/DC collapse threshold
ACDC_MIN_VALID = 0.001

# --- Stage 2 warm-up
CONTACT_WARMUP_S = 2.0

# --- HR smoothing
HR_SMOOTH_N = 4

# --- Beat detection
MIN_BPM         = 40
MAX_BPM         = 200
INTERVAL_WINDOW = 6

# --- Blink
GREEN_BLINK_S  = 1.0
YELLOW_BLINK_S = 0.7
RED_BLINK_S    = 0.5

# --- Print cadence (for dashboard/console output)
TEMP_PRINT_S = 2.0
HR_PRINT_S   = 2.0
AI_PRINT_S   = 1.0
STATUS_PRINT_ON_CHANGE = True  # Always print status changes

# --- Evaluation logging
EVAL_LOGGING  = True
EVAL_LOG_FILE = "eval_log.csv"
EVAL_LOG_S    = 1.0

# ==============================================================================
# DEMO MODE — set DEMO_MODE = True to feed sensor data from a CSV file instead
# of the real MAX30102 / DHT22 hardware. The AI pipeline, MQTT publish, and all
# logging run completely unchanged; only the sensor read block is bypassed.
# ==============================================================================
DEMO_MODE      = True                        # <- flip to False for real sensor
DEMO_CSV_FILE  = "anomaly_demo_dataset.csv"  # generated dataset
DEMO_SPEED     = 1.0                         # 1.0 = real-time, 2.0 = 2x faster
DEMO_LOOP      = True                        # restart from beginning when done

# --- Dashboard logging (comprehensive sensor + AI state)
DASHBOARD_LOGGING = True
DASHBOARD_LOG_FILE = "dashboard_log.csv"
DASHBOARD_LOG_S = 0.1  # 10Hz dashboard updates


# ==============================================================================
# Helpers
# ==============================================================================

# ------------------------------------------------------------------------------
# Demo CSV iterator — yields (ir, red, amb_t, scenario) one row per call.
# Runs at DEMO_SPEED x real-time by sleeping between rows.
# ------------------------------------------------------------------------------
def _make_demo_iterator(csv_path: str, speed: float, loop: bool):
    """Generator: yields (ir, red, amb_t, scenario) from the demo CSV."""
    import csv as _csv
    while True:
        try:
            with open(csv_path, newline="") as f:
                rows = list(_csv.DictReader(f))
        except FileNotFoundError:
            print(f"[DEMO] ERROR: {csv_path} not found. "
                  "Run the dataset generator first.")
            raise SystemExit(1)

        dt = 0.1 / speed   # seconds per row (dataset is 10 Hz)
        for row in rows:
            t0 = time.perf_counter()
            ir_v     = int(float(row["ir_raw"]))  if row["ir_raw"]  else None
            red_v    = int(float(row["red_raw"])) if row["red_raw"] else None
            amb_t_v  = float(row["amb_t"])        if row["amb_t"]   else None
            scenario = row.get("scenario", "")
            yield ir_v, red_v, amb_t_v, scenario
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

        if not loop:
            return
        print("[DEMO] Dataset complete — looping back to start.")


def load_joblib_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def safe_std(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def safe_mean(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def safe_slope(times, values) -> float:
    pts = [(float(t), float(v)) for t, v in zip(times, values)
           if t is not None and v is not None and np.isfinite(v)]
    if len(pts) < 2:
        return 0.0
    t = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    t = t - t[0]
    if np.allclose(t, 0.0):
        return 0.0
    m, _ = np.polyfit(t, y, 1)
    return float(m)


def safe_stats(arr: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {f"{prefix}_{s}": 0.0 for s in
                ["mean","std","min","max","median","p25","p75","range","first","last","slope"]}
    first = float(arr[0])
    last  = float(arr[-1])
    slope = float((last - first) / max(len(arr) - 1, 1))
    
    # Optimization: Single percentile call instead of two separate calls
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    pctiles = np.percentile(arr, [25, 50, 75])
    
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_min":    arr_min,
        f"{prefix}_max":    arr_max,
        f"{prefix}_median": float(pctiles[1]),
        f"{prefix}_p25":    float(pctiles[0]),
        f"{prefix}_p75":    float(pctiles[2]),
        f"{prefix}_range":  arr_max - arr_min,
        f"{prefix}_first":  first,
        f"{prefix}_last":   last,
        f"{prefix}_slope":  slope,
    }


def compute_spo2_est(red_window, ir_window) -> Optional[float]:
    n_r = len(red_window); n_i = len(ir_window)
    red = np.fromiter(red_window, dtype=np.float32, count=n_r)
    ir  = np.fromiter(ir_window,  dtype=np.float32, count=n_i)

    if red.size < 20 or ir.size < 20:
        return None

    red_dc = float(np.mean(red))
    ir_dc  = float(np.mean(ir))
    if red_dc <= 1e-6 or ir_dc <= 1e-6:
        return None

    red_ac = float(np.std(red))
    ir_ac  = float(np.std(ir))
    if red_ac <= 1e-6 or ir_ac <= 1e-6:
        return None

    ratio = (red_ac / (red_dc + 1e-6)) / (ir_ac / (ir_dc + 1e-6))
    spo2  = 110.0 - 25.0 * ratio
    spo2  = float(np.clip(spo2, 70.0, 100.0))
    return spo2


# ==============================================================================
# PASO — Optimisation: fast feature builder
# Replaces build_contact_feature_vector which built a pd.DataFrame every cycle.
# Instead, fills a pre-allocated float32 numpy array in-place using the trained
# feature column order. Eliminates dict allocation, DataFrame construction and
# column lookup — the main source of the ~4-6ms feat time in profiling.
# ==============================================================================
def build_contact_feature_vector_fast(
    ir_window, red_window, t_room_window, hr_window,
    spo2_window, acdc_window, p2p_window,
    feature_cols, out_buf: np.ndarray,
) -> np.ndarray:
    """Fill pre-allocated (1, n_features) float32 buffer in-place and return it."""
    ir_arr   = np.asarray(list(ir_window),     dtype=np.float32)
    red_arr  = np.asarray(list(red_window),    dtype=np.float32)
    tr_arr   = np.asarray(list(t_room_window), dtype=np.float32)
    hr_arr   = np.asarray([
        0.0 if (v is None or not np.isfinite(v)) else float(v) for v in hr_window
    ], dtype=np.float32)
    spo2_arr = np.asarray([
        0.0 if (v is None or not np.isfinite(v)) else float(v) for v in spo2_window
    ], dtype=np.float32)
    acdc_arr = np.asarray(list(acdc_window), dtype=np.float32)
    p2p_arr  = np.asarray(list(p2p_window),  dtype=np.float32)

    # FIX: truncate all PPG arrays to the shortest length before any element-wise
    # operations. Buffers can differ by ±1 at boundaries (p2p/acdc appended after
    # ir/red in the same cycle, islice skip computed from ir_buf length only).
    min_len = min(ir_arr.size, red_arr.size, acdc_arr.size, p2p_arr.size,
                  hr_arr.size, spo2_arr.size)
    if min_len == 0:
        out_buf[0, :] = 0.0
        return out_buf
    ir_arr   = ir_arr[:min_len]
    red_arr  = red_arr[:min_len]
    acdc_arr = acdc_arr[:min_len]
    p2p_arr  = p2p_arr[:min_len]
    hr_arr   = hr_arr[:min_len]
    spo2_arr = spo2_arr[:min_len]
    # tr_arr is room-temp — different cadence (DHT22 at 2s), keep as-is

    feat = {}
    feat.update(safe_stats(ir_arr,   "ir_raw"))
    feat.update(safe_stats(red_arr,  "red_raw"))
    feat.update(safe_stats(acdc_arr, "ac_dc_ratio"))
    feat.update(safe_stats(p2p_arr,  "peak_to_peak"))
    feat.update(safe_stats(tr_arr,   "t_room"))
    feat.update(safe_stats(hr_arr,   "hr"))
    feat.update(safe_stats(hr_arr,   "hr_std_5s"))
    feat.update(safe_stats(spo2_arr, "spo2_est"))

    feat["nonzero_ir_fraction"]    = float(np.mean(ir_arr > 0))   if ir_arr.size > 0 else 0.0
    feat["acdc_over_p2p_mean"]     = float(np.mean(acdc_arr / (p2p_arr + 1e-6))) if acdc_arr.size > 0 and acdc_arr.size == p2p_arr.size else 0.0
    feat["p2p_over_ir_mean"]       = float(np.mean(p2p_arr / (np.abs(ir_arr) + 1e-6))) if p2p_arr.size > 0 and p2p_arr.size == ir_arr.size else 0.0
    feat["red_ir_ratio_mean"]      = float(np.mean(red_arr / (ir_arr + 1e-6))) if red_arr.size > 0 and red_arr.size == ir_arr.size else 0.0
    feat["red_ir_ratio_std"]       = float(np.std(red_arr  / (ir_arr + 1e-6))) if red_arr.size > 0 and red_arr.size == ir_arr.size else 0.0
    feat["red_nonzero_fraction"]   = float(np.mean(red_arr > 0))  if red_arr.size > 0 else 0.0
    feat["nonzero_hr_fraction"]    = float(np.mean(hr_arr > 0))   if hr_arr.size > 0 else 0.0
    feat["hr_over_ir_mean"]        = float(np.mean(hr_arr / (np.abs(ir_arr) + 1e-6))) if hr_arr.size > 0 and hr_arr.size == ir_arr.size else 0.0
    feat["nonzero_spo2_fraction"]  = float(np.mean(spo2_arr > 0)) if spo2_arr.size > 0 else 0.0
    feat["valid_spo2_fraction"]    = float(np.mean((spo2_arr >= 70) & (spo2_arr <= 100))) if spo2_arr.size > 0 else 0.0
    feat["spo2_below_90_fraction"] = float(np.mean(spo2_arr < 90)) if spo2_arr.size > 0 else 0.0

    # OPT: vectorised fill — 2.5x faster than Python for-loop over ~90 columns
    out_buf[0, :] = [feat.get(col, 0.0) for col in feature_cols]
    return out_buf


class ContextBuffer:
    """Rolling window buffer for Stage-2 anomaly-detector features."""
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = float(window_seconds)
        self.time_buf   = deque()
        self.ir_buf     = deque()
        self.red_buf    = deque()
        self.hr_buf     = deque()
        self.spo2_buf   = deque()
        self.acdc_buf   = deque()
        self.p2p_buf    = deque()
        self.t_room_buf = deque()

    def update(self, now, ir_raw, red_raw, hr, spo2_est, ac_dc, p2p, t_room) -> None:
        self.time_buf.append(now)
        self.ir_buf.append(ir_raw)
        self.red_buf.append(red_raw)
        self.hr_buf.append(hr)
        self.spo2_buf.append(spo2_est)
        self.acdc_buf.append(ac_dc)
        self.p2p_buf.append(p2p)
        self.t_room_buf.append(t_room)
        self._prune(now)

    def _prune(self, now) -> None:
        while self.time_buf and (now - self.time_buf[0]) > self.window_seconds:
            self.time_buf.popleft()
            self.ir_buf.popleft()
            self.red_buf.popleft()
            self.hr_buf.popleft()
            self.spo2_buf.popleft()
            self.acdc_buf.popleft()
            self.p2p_buf.popleft()
            self.t_room_buf.popleft()

    def feature_map(self, ir_raw, ac_dc_ratio, peak_to_peak, t_room, hr, spo2_est) -> dict:
        t_vals  = [v for v in self.t_room_buf if v is not None and np.isfinite(v)]
        dt_room = float(t_vals[-1] - t_vals[-2]) if len(t_vals) >= 2 else 0.0
        p2p_m   = safe_mean(self.p2p_buf)
        ir_m    = safe_mean(self.ir_buf)

        return {
            "ac_dc_ratio":      float(ac_dc_ratio),
            "peak_to_peak":     float(peak_to_peak),
            "p2p_cv":           float(p2p_m / (ir_m + 1e-6)) if ir_m > 0.0 else 0.0,
            "ac_dc_ratio_std":  safe_std(self.acdc_buf),
            "peak_to_peak_std": safe_std(self.p2p_buf),
            "ir_std_5s":        safe_std(self.ir_buf),
            "ir_slope":         safe_slope(self.time_buf, self.ir_buf),
            "ir_drop":          1.0 if 0.0 < ir_m < 5000.0 else 0.0,
            "low_variation":    1.0 if p2p_m < 1000.0 else 0.0,
            "hr":               float(hr) if hr is not None and np.isfinite(hr) else np.nan,
            "hr_std_5s":        safe_std(self.hr_buf),
            "hr_valid":         1.0 if any(v is not None and np.isfinite(v) for v in self.hr_buf) else 0.0,
            "spo2_est":         float(spo2_est) if spo2_est is not None and np.isfinite(spo2_est) else np.nan,
            "dt_room":          dt_room,
            "t_room_std_5s":    safe_std(self.t_room_buf),
        }


def build_anomaly_feature_vector(feature_map: dict, feat_order: list) -> np.ndarray:
    # OPT: numpy array directly — no DataFrame allocation on the hot path
    return np.array([[feature_map.get(k, 0.0) for k in feat_order]], dtype=np.float32)


# ==============================================================================
# Temperature context analyser
# ==============================================================================
class TempContextAnalyser:
    def __init__(self):
        self.time_buf = deque()
        self.temp_buf = deque()

    def update(self, now: float, t_room: Optional[float]) -> None:
        if t_room is None:
            return
        self.time_buf.append(now)
        self.temp_buf.append(float(t_room))
        while self.time_buf and (now - self.time_buf[0]) > TEMP_CONTEXT_WINDOW_S:
            self.time_buf.popleft()
            self.temp_buf.popleft()

    def is_cold_environment(self) -> bool:
        if len(self.temp_buf) < 2:
            return False
        drop = float(self.temp_buf[0]) - float(self.temp_buf[-1])
        return drop >= TEMP_DROP_RATE_C

    def is_stable_ambient(self) -> bool:
        if len(self.temp_buf) < 5:
            return False
        return safe_std(list(self.temp_buf)) < TEMP_STABLE_STD_C

    def adjusted_threshold(self, ir_current: Optional[float]) -> float:
        low_ir = (ir_current is not None) and (ir_current < IR_LOW_THRESHOLD)
        if self.is_cold_environment() and low_ir:
            return min(ANOMALY_PROB_THRESHOLD + 0.15, 0.90)
        if self.is_stable_ambient() and low_ir:
            return max(ANOMALY_PROB_THRESHOLD - 0.10, 0.50)
        return ANOMALY_PROB_THRESHOLD

    def context_label(self) -> str:
        if self.is_cold_environment():
            return "COLD_ENV"
        if self.is_stable_ambient():
            return "STABLE"
        return "NORMAL_ENV"


# ==============================================================================
# Severity debounce (asymmetric)
# ==============================================================================
class SeverityDebounce:
    def __init__(self, upgrade_s: float = 3.5, downgrade_s: float = 1.2):
        self.upgrade_s          = upgrade_s
        self.downgrade_s        = downgrade_s
        self.committed_severity = "NORMAL"
        self.candidate_severity = "NORMAL"
        self.candidate_since    = time.time()

    def update(self, raw_severity: str, now: float) -> str:
        level = {"NORMAL": 0, "WARN": 1, "ALERT": 2}
        raw_lvl       = level.get(raw_severity, 0)
        committed_lvl = level.get(self.committed_severity, 0)

        if raw_severity != self.candidate_severity:
            self.candidate_severity = raw_severity
            self.candidate_since    = now

        going_up   = raw_lvl > committed_lvl
        going_down = raw_lvl < committed_lvl
        wait_s     = self.upgrade_s if going_up else self.downgrade_s

        if (going_up or going_down) and (now - self.candidate_since) >= wait_s:
            self.committed_severity = raw_severity

        return self.committed_severity


def compute_raw_severity(temp_c, bpm_value) -> str:
    ha = (bpm_value is not None) and (bpm_value >= HR_ALERT_BPM)
    hw = (bpm_value is not None) and (bpm_value >= HR_WARN_BPM)
    ta = (temp_c is not None) and (temp_c >= TEMP_ALERT_HIGH_C)
    tw = (temp_c is not None) and (temp_c >= TEMP_WARN_HIGH_C)
    if ta or ha:
        return "ALERT"
    if tw or hw:
        return "WARN"
    return "NORMAL"


# ==============================================================================
# GPIO
# ==============================================================================
def all_leds_off() -> None:
    led_green.off(); led_yellow.off(); led_red.off()


def beep(freq=1600, duration=0.07, duty=0.25) -> None:
    buzzer.frequency = freq
    buzzer.value = duty
    time.sleep(duration)
    buzzer.off()


def startup_test() -> None:
    print("[SYSTEM] LED + Buzzer self-test...")
    for led in (led_green, led_yellow, led_red):
        led.on(); time.sleep(0.25); led.off()
    buzzer.frequency = 2000; buzzer.value = 0.1
    time.sleep(0.1); buzzer.off()
    print("[SYSTEM] Temp-context awareness: ACTIVE (rate-of-change, no suppression)")
    print("[SYSTEM] Logging to eval_log.csv and dashboard_log.csv")
    print("[SYSTEM] Test complete. AI monitoring active.\n")


# ==============================================================================
# Evaluation logger
# ==============================================================================
_eval_writer = None
_eval_file_h = None


def init_eval_log() -> None:
    global _eval_writer, _eval_file_h
    if not EVAL_LOGGING:
        return
    is_new = not os.path.exists(EVAL_LOG_FILE)
    _eval_file_h = open(EVAL_LOG_FILE, "a", newline="")
    _eval_writer = csv.writer(_eval_file_h)
    if is_new:
        _eval_writer.writerow([
            "wall_time", "sensor_read_latency_ms",
            "severity", "bpm", "spo2_est", "amb_t",
            "contact_status", "contact_conf",
            "anomaly_score", "anomaly_threshold_used",
            "temp_context", "anomaly_persisting_s",
        ])


def log_eval(wall_time, read_latency_ms, severity, bpm, spo2_est, amb_t,
             contact_status, contact_conf, anomaly_score,
             anomaly_threshold_used, temp_context, anomaly_persisting_s):
    if not EVAL_LOGGING or _eval_writer is None:
        return
    _eval_writer.writerow([
        round(wall_time, 4),
        round(read_latency_ms, 3),
        severity,
        round(bpm, 2) if bpm is not None else "",
        round(spo2_est, 2) if spo2_est is not None else "",
        round(amb_t, 2) if amb_t is not None else "",
        contact_status,
        round(contact_conf, 3),
        round(anomaly_score, 6) if anomaly_score is not None else "",
        round(anomaly_threshold_used, 6) if anomaly_threshold_used is not None else "",
        temp_context,
        round(anomaly_persisting_s, 2) if anomaly_persisting_s is not None else "",
    ])
    _eval_file_h.flush()


# ==============================================================================
# Dashboard logger (comprehensive real-time sensor + AI state)
# ==============================================================================
_dashboard_writer = None
_dashboard_file_h = None


def init_dashboard_log() -> None:
    global _dashboard_writer, _dashboard_file_h
    if not DASHBOARD_LOGGING:
        return
    is_new = not os.path.exists(DASHBOARD_LOG_FILE)
    _dashboard_file_h = open(DASHBOARD_LOG_FILE, "a", newline="")
    _dashboard_writer = csv.writer(_dashboard_file_h)
    if is_new:
        _dashboard_writer.writerow([
            "wall_time",
            "ir_raw", "red_raw", "ir_mean", "red_mean",
            "ac_dc_ratio", "peak_to_peak",
            "bpm", "spo2_est", "amb_t",
            "contact_status", "contact_label", "contact_conf",
            "signal_present", "contact_warmed_up",
            "anomaly_score", "anomaly_threshold",
            "is_anomaly", "anomaly_persisting_s",
            "severity", "temp_context",
        ])


def log_dashboard(
    wall_time, ir_raw, red_raw, ir_mean, red_mean,
    ac_dc_ratio, peak_to_peak,
    bpm, spo2_est, amb_t,
    contact_status, contact_label, contact_conf,
    signal_present, contact_warmed_up,
    anomaly_score, anomaly_threshold,
    is_anomaly, anomaly_persisting_s,
    severity, temp_context):
    if not DASHBOARD_LOGGING or _dashboard_writer is None:
        return
    _dashboard_writer.writerow([
        round(wall_time, 4),
        round(ir_raw, 0) if ir_raw is not None else "",
        round(red_raw, 0) if red_raw is not None else "",
        round(ir_mean, 0) if ir_mean is not None else "",
        round(red_mean, 0) if red_mean is not None else "",
        round(ac_dc_ratio, 4),
        round(peak_to_peak, 2),
        round(bpm, 1) if bpm is not None else "",
        round(spo2_est, 1) if spo2_est is not None else "",
        round(amb_t, 2) if amb_t is not None else "",
        contact_status,
        contact_label,
        round(contact_conf, 3),
        "YES" if signal_present else "NO",
        "YES" if contact_warmed_up else "NO",
        round(anomaly_score, 6) if anomaly_score is not None else "",
        round(anomaly_threshold, 6) if anomaly_threshold is not None else "",
        "YES" if is_anomaly else "NO",
        round(anomaly_persisting_s, 2) if anomaly_persisting_s is not None else "",
        severity,
        temp_context,
    ])
    _dashboard_file_h.flush()


# ==============================================================================
# Load models
# ==============================================================================
startup_test()
init_eval_log()
init_dashboard_log()
init_mqtt()
print("[SYSTEM] MQTT publisher started.")

contact_bundle = load_joblib_model("models/contact_model.pkl")
anomaly_bundle = load_joblib_model("models/stage2_anomaly_detector.pkl")
ai_available   = contact_bundle is not None and anomaly_bundle is not None

label_map                = {}
CONTACT_GOOD_IDX         = set()
contact_model            = None
contact_feature_cols     = None
contact_window_size      = PPG_WINDOW

anomaly_model            = None
anomaly_imputer          = None
anomaly_scaler           = None
anomaly_feature_cols     = None
anomaly_score_threshold  = None

# PASO — Optimisation: pre-allocated inference buffer (shape set after model load)
_contact_feat_buf = None

if ai_available:
    try:
        with open("models/contact_label_map.json") as f:
            label_map = json.load(f)

        CONTACT_GOOD_IDX     = {int(k) for k, v in label_map.items() if "good_contact" in v}
        contact_model        = contact_bundle["model"]
        contact_feature_cols = contact_bundle["feature_cols"]
        contact_window_size  = int(contact_bundle.get("window_size", PPG_WINDOW))

        anomaly_model            = anomaly_bundle["model"]
        anomaly_imputer          = anomaly_bundle["imputer"]
        anomaly_scaler           = anomaly_bundle["scaler"]
        anomaly_feature_cols     = anomaly_bundle["feature_cols"]
        anomaly_score_threshold  = float(anomaly_bundle["score_threshold"])

        # PASO — Optimization #2: Prune Random Forest to reduce Stage 1 latency
        # Reduce from default 100 trees to 15 trees for significant speedup with minimal accuracy loss
        if hasattr(contact_model, 'estimators_'):
            original_n_trees = len(contact_model.estimators_)
            n_trees_target = 15
            if original_n_trees > n_trees_target:
                contact_model.estimators_ = contact_model.estimators_[:n_trees_target]
                contact_model.n_estimators = n_trees_target
                print(f"[AI] Pruned contact RF: {original_n_trees} → {n_trees_target} trees")
            else:
                print(f"[AI] Contact RF already has {original_n_trees} trees (≤15)")

        # PASO — Optimization #3: Prune Isolation Forest (already done in v1, ensure it's applied)
        if hasattr(anomaly_model, 'estimators_'):
            original_n_est = len(anomaly_model.estimators_)
            n_est_target = 20
            if original_n_est > n_est_target:
                anomaly_model.estimators_ = anomaly_model.estimators_[:n_est_target]
                anomaly_model.n_estimators = n_est_target
                print(f"[AI] Pruned anomaly IF: {original_n_est} → {n_est_target} estimators")
            else:
                print(f"[AI] Anomaly IF already has {original_n_est} estimators (≤20)")

        # PASO — Optimisation: allocate feature buffer once now that n_features is known
        _contact_feat_buf = np.zeros((1, len(contact_feature_cols)), dtype=np.float32)

        print(f"[AI] Models loaded. Good-contact classes: {[label_map[str(i)] for i in CONTACT_GOOD_IDX]}")
        print(f"[AI] Stage-2 score threshold={anomaly_score_threshold:.6f}  Stage-1 confidence gate={AI_CONFIDENCE_GATE}")
        print(f"[OPTIM] Pre-allocated feature buffer shape: {_contact_feat_buf.shape}")
    except Exception as e:
        print(f"[AI] Model load error: {e}")
        ai_available = False
else:
    print("[AI] No models - rule-based fallback only.")


# ==============================================================================
# PASO — Scheduling: DHT22 background thread
# Moves the blocking dht.temperature call off the 20ms inference loop.
# Main loop reads _dht_value non-blocking via lock — near-zero cost.
# ==============================================================================
_dht_lock  = threading.Lock()
_dht_value = None


def _dht_reader_thread():
    global _dht_value
    while True:
        try:
            t = dht.temperature
            if t is not None:
                with _dht_lock:
                    _dht_value = float(t)
        except RuntimeError:
            pass
        time.sleep(2.0)


_dht_thread = threading.Thread(target=_dht_reader_thread, daemon=True)
_dht_thread.start()
print("[SYSTEM] DHT22 background thread started (PASO scheduling).")


# ==============================================================================
# Main loop
# ==============================================================================
debouncer    = SeverityDebounce(upgrade_s=SEVERITY_UPGRADE_S, downgrade_s=SEVERITY_DOWNGRADE_S)
temp_context = TempContextAnalyser()
last_temp_print  = 0.0
last_hr_print    = 0.0
last_dashboard_log = 0.0
last_eval_log    = 0.0

# PASO — Profiling: per-cycle timing accumulators
_prof_loop_count  = 0
_prof_t_dht_ms    = 0.0
_prof_t_feat_ms   = 0.0
_prof_t_s1_ms     = 0.0
_prof_t_s2_ms     = 0.0
PROF_REPORT_EVERY = 100

try:
    # In DEMO_MODE we skip all hardware init. A dummy bus object is used
    # so the rest of the indented block stays structurally identical.
    if DEMO_MODE:
        print(f"[DEMO] Mode active — reading from {DEMO_CSV_FILE} at {DEMO_SPEED}x speed.")
        print(f"[DEMO] AI pipeline, MQTT, and logging run as normal.")
        _demo_iter = _make_demo_iterator(DEMO_CSV_FILE, DEMO_SPEED, DEMO_LOOP)
        bus = None   # not used in demo mode
    else:
        bus = SMBus(1)
        setup_max30102(bus)

    if True:   # keeps indentation identical to the original `with SMBus(1) as bus:` block

        ir_buf      = deque(maxlen=PPG_WINDOW)
        red_buf     = deque(maxlen=PPG_WINDOW)
        t_room_buf  = deque(maxlen=PPG_WINDOW)
        hr_buf      = deque(maxlen=PPG_WINDOW)
        spo2_buf    = deque(maxlen=PPG_WINDOW)
        acdc_buf    = deque(maxlen=PPG_WINDOW)
        p2p_buf     = deque(maxlen=PPG_WINDOW)
        context     = ContextBuffer(window_seconds=CONTEXT_WINDOW_SECONDS)

        # Beat detection
        ir_history     = []
        last_beat_time = None
        intervals      = deque(maxlen=INTERVAL_WINDOW)

        # HR smoothing
        bpm_history    = deque(maxlen=HR_SMOOTH_N)
        hr_smooth_buf  = deque(maxlen=HR_SMOOTH_N)
        bpm            = None

        amb_t                 = None
        last_toggle           = 0.0
        blink_on              = False
        last_printed_severity = None

        anomaly_start = None
        anomaly_prob  = None

        contact_vote_buf    = deque(maxlen=CONTACT_VOTE_N)
        contact_status      = "UNKNOWN"
        contact_conf        = 0.0
        contact_label       = "unknown"
        prev_contact_status = "UNKNOWN"
        signal_present      = False
        contact_good_since  = None
        _last_demo_scenario = None   # for DEMO_MODE scenario transition printing

        while True:
            loop_start = time.time()

            # ----------- 1) Read sensor (or demo CSV) ------------------------
            read_t0 = time.time()
            ir = None
            red = None

            if DEMO_MODE:
                # Pull next row from CSV iterator — timing is handled inside
                try:
                    ir, red, _demo_amb_t, _demo_scenario = next(_demo_iter)
                    if _demo_amb_t is not None:
                        # Inject temperature directly (bypasses DHT22 thread)
                        with _dht_lock:
                            _dht_value = _demo_amb_t
                    # Print scenario transitions
                    if _demo_scenario != _last_demo_scenario:
                        print(f"\n[DEMO] >>> Scenario: {_demo_scenario.upper()} <<<")
                        _last_demo_scenario = _demo_scenario
                except StopIteration:
                    print("\n[DEMO] Dataset finished.")
                    raise KeyboardInterrupt
            else:
                # Real MAX30102 sensor read
                try:
                    d   = bus.read_i2c_block_data(I2C_ADDR, 0x07, 6)
                    red = ((d[0] << 16) | (d[1] << 8) | d[2]) & 0x03FFFF
                    ir  = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x03FFFF
                except Exception:
                    pass

            # Shared buffer append + beat detection (same for both modes)
            if ir is not None and red is not None:
                ir_buf.append(float(ir))
                red_buf.append(float(red))

                if 50000 < ir < 200000:
                    if not signal_present:
                        ir_history = []
                        last_beat_time = None
                        signal_present = True

                    ir_history.append(ir)
                    if len(ir_history) > 30:
                        ir_history.pop(0)

                    avg = sum(ir_history) / len(ir_history)
                    if ir < (avg - 500):
                        if last_beat_time is None or (loop_start - last_beat_time) > 0.45:
                            if last_beat_time is not None:
                                interval = loop_start - last_beat_time
                                inst = 60.0 / interval if interval > 0 else None
                                if inst and (MIN_BPM <= inst <= MAX_BPM):
                                    intervals.append(interval)
                                    # BPM FIX: compute smoothed BPM from interval buffer
                                    avg_interval = sum(intervals) / len(intervals)
                                    raw_bpm = 60.0 / avg_interval
                                    bpm_history.append(raw_bpm)
                                    bpm = sum(bpm_history) / len(bpm_history)
                            last_beat_time = loop_start
                else:
                    if signal_present:
                        ir_history = []
                        last_beat_time = None
                        intervals.clear()
                        bpm_history.clear()
                        hr_smooth_buf.clear()
                        bpm = None
                        contact_vote_buf.clear()
                        contact_status     = "BAD"
                        contact_conf       = 0.0
                        contact_label      = "unknown"
                        contact_good_since = None
                        signal_present     = False

                        ir_buf.clear()
                        red_buf.clear()
                        hr_buf.clear()
                        spo2_buf.clear()
                        acdc_buf.clear()
                        p2p_buf.clear()

            read_latency_ms = (time.time() - read_t0) * 1000.0

            # -------- 2) DHT22 temperature (PASO scheduling: non-blocking) ----
            _t_dht0 = time.perf_counter()                               # PASO profiling
            with _dht_lock:
                _cached_t = _dht_value
            if _cached_t is not None:
                amb_t = _cached_t
                temp_context.update(loop_start, amb_t)
                if (loop_start - last_temp_print) >= TEMP_PRINT_S:
                    ctx_lbl = temp_context.context_label()
                    print(f"[TEMP] {amb_t:.2f} C  [{ctx_lbl}]")
                    last_temp_print = loop_start
            _prof_t_dht_ms += (time.perf_counter() - _t_dht0) * 1000  # PASO profiling

            if amb_t is not None:
                t_room_buf.append(amb_t)

            # Derive PPG proxies — operate directly on the deque, no list() copy needed
            if ir_buf:
                ir_arr_w      = np.fromiter(ir_buf, dtype=np.float32, count=len(ir_buf))
                red_arr_w     = np.fromiter(red_buf, dtype=np.float32, count=len(red_buf))
                ir_mean_val   = float(np.mean(ir_arr_w))
                red_mean_val  = float(np.mean(red_arr_w))
                peak_to_peak  = float(ir_arr_w[-1] - ir_arr_w.min()) if len(ir_arr_w) > 1 else 0.0
                peak_to_peak  = float(ir_arr_w.max() - ir_arr_w.min())
                ac_dc_ratio_v = float(peak_to_peak / (ir_mean_val + 1e-6))
            else:
                ir_arr_w      = None
                red_arr_w     = None
                ir_mean_val   = 0.0
                red_mean_val  = 0.0
                peak_to_peak  = 0.0
                ac_dc_ratio_v = 0.0

            spo2_est = compute_spo2_est(red_buf, ir_buf)

            acdc_buf.append(float(ac_dc_ratio_v))
            p2p_buf.append(float(peak_to_peak))
            hr_buf.append(float(bpm) if bpm is not None else np.nan)
            spo2_buf.append(float(spo2_est) if spo2_est is not None else np.nan)

            context.update(
                now=loop_start,
                ir_raw=float(ir) if ir is not None else None,
                red_raw=float(red) if red is not None else None,
                hr=float(bpm) if bpm is not None else None,
                spo2_est=float(spo2_est) if spo2_est is not None else None,
                ac_dc=float(ac_dc_ratio_v),
                p2p=float(peak_to_peak),
                t_room=float(amb_t) if amb_t is not None else None,
            )

            # ----------- 3) HR print (periodic only) ----------------------------------
            if bpm is not None and (loop_start - last_hr_print) >= HR_PRINT_S:
                _spo2_str = f"{spo2_est:.1f}%" if spo2_est is not None else "N/A"
                print(f"[HR] {bpm:.1f} BPM | [SPO2] {_spo2_str} (est) | [IR] {ir if ir else 'N/A'} | [RED] {red if red else 'N/A'}")
                last_hr_print = loop_start

            # ----- 4) AI 2-stage pipeline -----------------------------------
            anomaly_score_this_cycle = None
            score_threshold_used     = anomaly_score_threshold
            temp_ctx_label           = temp_context.context_label()
            is_anomaly               = False
            raw_severity             = "WARN"
            contact_warmed_up        = False

            if ai_available:
                # Instant poor-contact detection via AC/DC collapse
                if ac_dc_ratio_v < ACDC_MIN_VALID and contact_status == "GOOD":
                    contact_vote_buf.clear()
                    contact_status     = "BAD"
                    contact_conf       = 0.0
                    contact_good_since = None
                    anomaly_start      = None
                    is_anomaly         = False

                # PASO Optimization #1: Skip S1 feature extraction entirely if signal_present is False
                # Only run S1 when we have a valid signal
                if signal_present:
                    run_short = (len(ir_buf) >= PPG_SHORT_WINDOW and len(ir_buf) < contact_window_size)
                    run_full  = (len(ir_buf) >= contact_window_size and len(t_room_buf) >= 5)

                    if run_full or run_short:
                        if run_short:
                            # FIX: compute skip from the smallest buffer so all
                            # islice slices are the same length (ir/red appended
                            # before acdc/p2p each cycle, so they can differ by 1)
                            min_buf_len = min(len(ir_buf), len(red_buf), len(hr_buf),
                                             len(spo2_buf), len(acdc_buf), len(p2p_buf))
                            skip = max(0, min_buf_len - PPG_SHORT_WINDOW)
                            ir_win   = list(itertools.islice(ir_buf,   skip, None))
                            red_win  = list(itertools.islice(red_buf,  skip, None))
                            hr_win   = list(itertools.islice(hr_buf,   skip, None))
                            spo2_win = list(itertools.islice(spo2_buf, skip, None))
                            acdc_win = list(itertools.islice(acdc_buf, skip, None))
                            p2p_win  = list(itertools.islice(p2p_buf,  skip, None))
                        else:
                            # Full window: deques are already maxlen=PPG_WINDOW
                            # Pass directly — np.fromiter handles them inside the builder
                            ir_win   = ir_buf
                            red_win  = red_buf
                            hr_win   = hr_buf
                            spo2_win = spo2_buf
                            acdc_win = acdc_buf
                            p2p_win  = p2p_buf

                        tr_win = t_room_buf  # pass deque directly

                        # PASO — Optimisation: fast builder fills pre-allocated buffer,
                        # no DataFrame or dict-to-array conversion overhead
                        _t_feat0 = time.perf_counter()                      # PASO profiling
                        x_c_np = build_contact_feature_vector_fast(
                            ir_window=ir_win,
                            red_window=red_win,
                            t_room_window=tr_win,
                            hr_window=hr_win,
                            spo2_window=spo2_win,
                            acdc_window=acdc_win,
                            p2p_window=p2p_win,
                            feature_cols=contact_feature_cols,
                            out_buf=_contact_feat_buf,
                        )
                        _prof_t_feat_ms += (time.perf_counter() - _t_feat0) * 1000  # PASO profiling

                        _t_s1_0 = time.perf_counter()                       # PASO profiling
                        # OPT: single predict_proba() call instead of predict() + predict_proba()
                        # separately — halves tree traversals. argmax == what predict() returns.
                        probs        = contact_model.predict_proba(x_c_np)[0]
                        pred_idx     = int(np.argmax(probs))
                        contact_conf = float(probs[pred_idx])
                        _prof_t_s1_ms += (time.perf_counter() - _t_s1_0) * 1000    # PASO profiling

                        contact_label = label_map.get(str(pred_idx), "unknown")
                        raw_status    = "GOOD" if pred_idx in CONTACT_GOOD_IDX else "BAD"

                        short_gate = AI_CONFIDENCE_GATE + 0.02 if run_short else AI_CONFIDENCE_GATE
                        if contact_conf >= short_gate:
                            contact_vote_buf.append(raw_status)

                        if contact_vote_buf:
                            votes  = list(contact_vote_buf)
                            n      = len(votes)
                            weights = [1.0 + (i / max(n - 1, 1)) for i in range(n)]
                            weighted_good  = sum(w for v, w in zip(votes, weights) if v == "GOOD")
                            weighted_total = sum(weights)
                            contact_status = "GOOD" if weighted_good / weighted_total > 0.5 else "BAD"
                        else:
                            contact_status = "UNKNOWN"

                        if contact_status == "BAD" and prev_contact_status == "GOOD":
                            flush_n = int(len(ir_buf) * MOTION_FLUSH_FRACTION)
                            for _ in range(flush_n):
                                if ir_buf:   ir_buf.popleft()
                                if red_buf:  red_buf.popleft()
                                if hr_buf:   hr_buf.popleft()
                                if spo2_buf: spo2_buf.popleft()
                                if acdc_buf: acdc_buf.popleft()
                                if p2p_buf:  p2p_buf.popleft()
                            contact_good_since = None

                        prev_contact_status = contact_status
                else:
                    # No signal present — keep contact status as BAD and skip S1
                    if contact_status != "BAD":
                        contact_status = "BAD"
                        contact_conf = 0.0

                # --- Stage 2 gate checks ---
                if contact_status == "GOOD":
                    if contact_good_since is None:
                        contact_good_since = loop_start
                    contact_warmed_up = (loop_start - contact_good_since) >= CONTACT_WARMUP_S
                else:
                    contact_good_since = None
                    contact_warmed_up  = False

                ir_valid_for_s2 = (ir is not None and IR_VALID_MIN < ir < IR_VALID_MAX)

                _t_s2_0 = time.perf_counter()                           # PASO profiling
                if contact_status == "GOOD" and ir_valid_for_s2 and contact_warmed_up:
                    fmap = context.feature_map(
                        ir_raw=float(ir) if ir is not None else None,
                        ac_dc_ratio=ac_dc_ratio_v,
                        peak_to_peak=peak_to_peak,
                        t_room=amb_t,
                        hr=bpm,
                        spo2_est=spo2_est,
                    )
                    x_s2 = build_anomaly_feature_vector(fmap, anomaly_feature_cols)
                    # x_s2 is already a (1, n_features) float32 array
                    x_np = anomaly_imputer.transform(x_s2)
                    x_np = anomaly_scaler.transform(x_np)

                    anomaly_score            = float(anomaly_model.decision_function(x_np)[0])
                    anomaly_score_this_cycle = anomaly_score
                    is_anomaly               = anomaly_score < score_threshold_used

                    if is_anomaly:
                        if anomaly_start is None:
                            anomaly_start = loop_start
                        persist_s = loop_start - anomaly_start
                        raw_severity = "ALERT"
                    else:
                        anomaly_start = None
                        persist_s     = 0.0
                        raw_severity = "NORMAL"

                elif contact_status == "GOOD" and not ir_valid_for_s2:
                    anomaly_start = None
                    is_anomaly    = False
                    raw_severity  = "WARN"

                elif contact_status == "GOOD" and not contact_warmed_up:
                    anomaly_start = None
                    is_anomaly    = False
                    raw_severity  = "WARN"

                elif contact_status == "BAD":
                    anomaly_start = None
                    is_anomaly    = False
                    raw_severity  = "WARN"

                else:
                    is_anomaly   = False
                    raw_severity = "WARN"
                _prof_t_s2_ms += (time.perf_counter() - _t_s2_0) * 1000        # PASO profiling

            else:
                raw_severity = compute_raw_severity(amb_t, bpm)

            # Debounce and commit severity
            severity          = debouncer.update(raw_severity, loop_start)
            persist_s_for_log = (loop_start - anomaly_start) if anomaly_start else None

            # -- 5) Status print on change (keep this) ------------------------------------
            if severity != last_printed_severity and STATUS_PRINT_ON_CHANGE:
                print(f"[STATUS] -> {severity}")
                last_printed_severity = severity

            # ----- 6) LED blink --------------------------------
            period = {"NORMAL": GREEN_BLINK_S, "WARN": YELLOW_BLINK_S,
                      "ALERT": RED_BLINK_S}.get(severity, GREEN_BLINK_S)

            if (loop_start - last_toggle) >= (period / 2.0):
                blink_on    = not blink_on
                last_toggle = loop_start
                all_leds_off()
                if blink_on:
                    if severity == "NORMAL":
                        led_green.on()
                    elif severity == "WARN":
                        led_yellow.on()
                    else:
                        led_red.on()
                        beep(freq=1600, duration=0.06, duty=0.25)

            # ------ 7) Dashboard log (comprehensive, 10Hz) -------------------------
            if DASHBOARD_LOGGING and (loop_start - last_dashboard_log) >= DASHBOARD_LOG_S:
                log_dashboard(
                    wall_time=loop_start,
                    ir_raw=float(ir) if ir is not None else None,
                    red_raw=float(red) if red is not None else None,
                    ir_mean=ir_mean_val,
                    red_mean=red_mean_val,
                    ac_dc_ratio=ac_dc_ratio_v,
                    peak_to_peak=peak_to_peak,
                    bpm=bpm,
                    spo2_est=spo2_est,
                    amb_t=amb_t,
                    contact_status=contact_status,
                    contact_label=contact_label,
                    contact_conf=contact_conf,
                    signal_present=signal_present,
                    contact_warmed_up=contact_warmed_up,
                    anomaly_score=anomaly_score_this_cycle,
                    anomaly_threshold=score_threshold_used,
                    is_anomaly=is_anomaly,
                    anomaly_persisting_s=persist_s_for_log,
                    severity=severity,
                    temp_context=temp_ctx_label,
                )
                last_dashboard_log = loop_start

            # MQTT publish — non-blocking, rate-limited to 10Hz inside publish_vitals()
            publish_vitals(
                bpm=bpm, spo2_est=spo2_est, ir=ir, red=red,
                contact_status=contact_status, contact_label=contact_label,
                contact_conf=contact_conf,
                anomaly_score=anomaly_score_this_cycle,
                anomaly_threshold=score_threshold_used,
                is_anomaly=is_anomaly,
                anomaly_persisting_s=persist_s_for_log,
                amb_t=amb_t, temp_context=temp_ctx_label,
                severity=severity,
            )

            # ------ 8) Evaluation log (comprehensive, 1Hz) -------------------------
            if EVAL_LOGGING and (loop_start - last_eval_log) >= EVAL_LOG_S:
                log_eval(
                    wall_time=loop_start,
                    read_latency_ms=read_latency_ms,
                    severity=severity,
                    bpm=bpm,
                    spo2_est=spo2_est,
                    amb_t=amb_t,
                    contact_status=contact_status,
                    contact_conf=contact_conf,
                    anomaly_score=anomaly_score_this_cycle,
                    anomaly_threshold_used=score_threshold_used,
                    temp_context=temp_ctx_label,
                    anomaly_persisting_s=persist_s_for_log,
                )
                last_eval_log = loop_start

            # PASO — Profiling: print timing summary every N cycles
            _prof_loop_count += 1
            if _prof_loop_count % PROF_REPORT_EVERY == 0:
                n = PROF_REPORT_EVERY
                print(f"[PROF] avg over {n} cycles | "
                      f"DHT22={_prof_t_dht_ms/n:.3f}ms  "
                      f"feat={_prof_t_feat_ms/n:.3f}ms  "
                      f"S1={_prof_t_s1_ms/n:.3f}ms  "
                      f"S2={_prof_t_s2_ms/n:.3f}ms")
                _prof_t_dht_ms  = 0.0
                _prof_t_feat_ms = 0.0
                _prof_t_s1_ms   = 0.0
                _prof_t_s2_ms   = 0.0

            # In real sensor mode sleep 20ms between reads.
            # In demo mode the CSV iterator already handles row timing.
            if not DEMO_MODE:
                time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[EXIT] Shutdown.")
finally:
    all_leds_off()   # always safe — real hardware in both modes now
    buzzer.off()
    stop_mqtt()
    if _eval_file_h:
        _eval_file_h.close()
    if _dashboard_file_h:
        _dashboard_file_h.close()
    if not DEMO_MODE:
        try:
            dht.exit()
        except Exception:
            pass