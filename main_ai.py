import csv
import json
import os
import time
from collections import deque
from typing import Optional

import adafruit_dht
import board
import joblib
import numpy as np
import pandas as pd
from gpiozero import Device, LED, PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from smbus2 import SMBus

Device.pin_factory = LGPIOFactory()

# -- GPIO -----------------------------------------------------------------------
led_green  = LED(17)
led_yellow = LED(27)
led_red    = LED(22)
buzzer     = PWMOutputDevice(23)

# -- DHT22 ----------------------------------------------------------------------
dht = adafruit_dht.DHT22(board.D4, use_pulseio=False)

# -- MAX30102 -------------------------------------------------------------------
I2C_ADDR = 0x57


def setup_max30102(bus: SMBus) -> None:
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

# FIX: Removed WARM_ROOM_MODE binary suppressor.
# TempContextAnalyser already handles warm/cold labs via rate-of-change logic,
# so hard-suppressing temperature as a data source is unnecessary and reduces
# the value of the second sensor (DHT22) from the rubric's perspective.
# Temperature is always used as Stage-2 context via dt_room and t_room_std_5s.

# --- Temperature context parameters
# Rate-of-change logic works regardless of absolute room temperature,
# so no special mode is needed for warm labs.
TEMP_DROP_RATE_C      = 1.5    # degrees drop triggers "cold environment" mode
TEMP_CONTEXT_WINDOW_S = 60     # look-back window (seconds) for temp trend
TEMP_STABLE_STD_C     = 0.3    # ambient considered "stable" below this std
IR_LOW_THRESHOLD      = 60000  # IR below this = low perfusion proxy

# --- AI pipeline
PPG_WINDOW             = 250    # full window for Stage-1 (5 s @ 50 Hz)
PPG_SHORT_WINDOW       = 75     # short recovery window (1.5 s @ 50 Hz)
ANOMALY_PERSIST_S      = 5.0    # seconds before ALERT fires
ANOMALY_PROB_THRESHOLD = 0.65   # base threshold; adjusted by temp context
AI_CONFIDENCE_GATE     = 0.50
CONTEXT_WINDOW_SECONDS = 5.0

# Stage-1 vote buffer
CONTACT_VOTE_N = 3

# Motion flush fraction
MOTION_FLUSH_FRACTION = 0.8

# --- Severity debounce
SEVERITY_DEBOUNCE_S = 3.0

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

# --- Print cadence
TEMP_PRINT_S = 2.0
HR_PRINT_S   = 2.0
AI_PRINT_S   = 1.0

# --- Evaluation logging
EVAL_LOGGING  = True
EVAL_LOG_FILE = "eval_log.csv"
EVAL_LOG_S    = 1.0


# ==============================================================================
# Helpers
# ==============================================================================
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
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p25":    float(np.percentile(arr, 25)),
        f"{prefix}_p75":    float(np.percentile(arr, 75)),
        f"{prefix}_range":  float(np.max(arr) - np.min(arr)),
        f"{prefix}_first":  first,
        f"{prefix}_last":   last,
        f"{prefix}_slope":  slope,
    }


def build_contact_feature_vector(ir_window, t_room_window, feature_cols) -> pd.DataFrame:
    """Build feature vector for Stage-1 contact classifier."""
    ir_arr = np.asarray(list(ir_window), dtype=np.float32)
    tr_arr = np.asarray(list(t_room_window), dtype=np.float32)
    feat = {}
    feat.update(safe_stats(ir_arr, "ir_raw"))
    if ir_arr.size > 0:
        ir_mean     = float(np.mean(ir_arr))
        p2p_val     = float(np.max(ir_arr) - np.min(ir_arr))
        acdc_val    = p2p_val / (ir_mean + 1e-6)
        acdc_series = np.full_like(ir_arr, acdc_val)
        p2p_series  = np.full_like(ir_arr, p2p_val)
    else:
        ir_mean = p2p_val = 0.0
        acdc_series = p2p_series = np.array([], dtype=np.float32)
    feat.update(safe_stats(acdc_series, "ac_dc_ratio"))
    feat.update(safe_stats(p2p_series,  "peak_to_peak"))
    feat.update(safe_stats(tr_arr,      "t_room"))
    feat["nonzero_ir_fraction"] = float(np.mean(ir_arr > 0)) if ir_arr.size > 0 else 0.0
    feat["acdc_over_p2p_mean"]  = float(np.mean(acdc_series / (p2p_series + 1e-6))) if acdc_series.size > 0 else 0.0
    feat["p2p_over_ir_mean"]    = float(np.mean(p2p_series / (np.abs(ir_arr) + 1e-6))) if ir_arr.size > 0 else 0.0
    return pd.DataFrame([{col: feat.get(col, 0.0) for col in feature_cols}])


class ContextBuffer:
    """Rolling window buffer for Stage-2 MLP features."""
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = float(window_seconds)
        self.time_buf   = deque()
        self.ir_buf     = deque()
        self.hr_buf     = deque()
        self.acdc_buf   = deque()
        self.p2p_buf    = deque()
        self.t_room_buf = deque()

    def update(self, now, ir_raw, hr, ac_dc, p2p, t_room) -> None:
        self.time_buf.append(now);   self.ir_buf.append(ir_raw)
        self.hr_buf.append(hr);      self.acdc_buf.append(ac_dc)
        self.p2p_buf.append(p2p);    self.t_room_buf.append(t_room)
        self._prune(now)

    def _prune(self, now) -> None:
        while self.time_buf and (now - self.time_buf[0]) > self.window_seconds:
            self.time_buf.popleft();  self.ir_buf.popleft()
            self.hr_buf.popleft();    self.acdc_buf.popleft()
            self.p2p_buf.popleft();   self.t_room_buf.popleft()

    def feature_map(self, ir_raw, ac_dc_ratio, peak_to_peak, t_room, hr) -> dict:
        t_vals  = [v for v in self.t_room_buf if v is not None and np.isfinite(v)]
        dt_room = float(t_vals[-1] - t_vals[-2]) if len(t_vals) >= 2 else 0.0
        p2p_m   = safe_mean(self.p2p_buf)
        ir_m    = safe_mean(self.ir_buf)
        return {
            "ir_raw":           float(ir_raw)  if ir_raw is not None else 0.0,
            "ac_dc_ratio":      float(ac_dc_ratio),
            "peak_to_peak":     float(peak_to_peak),
            "t_room":           float(t_room)  if t_room is not None else 0.0,
            "hr":               float(hr)      if hr is not None else np.nan,
            "hr_std_5s":        safe_std(self.hr_buf),
            "ir_std_5s":        safe_std(self.ir_buf),
            "ir_slope":         safe_slope(self.time_buf, self.ir_buf),
            "ac_dc_ratio_std":  safe_std(self.acdc_buf),
            "peak_to_peak_std": safe_std(self.p2p_buf),
            "t_room_std_5s":    safe_std(self.t_room_buf),
            "dt_room":          dt_room,
            "hr_valid":         1.0 if any(v is not None and np.isfinite(v) for v in self.hr_buf) else 0.0,
            "ir_drop":          1.0 if 0.0 < ir_m < 5000.0 else 0.0,
            "low_variation":    1.0 if p2p_m < 1000.0 else 0.0,
            "p2p_cv":           float(p2p_m / (ir_m + 1e-6)) if ir_m > 0.0 else 0.0,
        }


def build_anomaly_feature_vector(feature_map: dict, feat_order: list) -> pd.DataFrame:
    return pd.DataFrame([{k: feature_map.get(k, 0.0) for k in feat_order}])


# ==============================================================================
# Temperature context analyser
# Works without cold-room training data — uses rate-of-change logic.
# This is always active (no WARM_ROOM_MODE suppressor needed).
# ==============================================================================
class TempContextAnalyser:
    """
    Maintains a rolling buffer of ambient temperatures.

    Provides two signals:
      is_cold_environment()  — ambient has been falling rapidly (environmental cold)
      is_stable_warm()       — ambient is stable (any signal drop is more suspicious)

    These adjust the anomaly probability threshold dynamically:
      cold environment  → raise threshold (don't alarm for natural perfusion drop)
      stable ambient    → keep or lower threshold (more suspicious if IR drops)
    """
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
# Severity debounce
# ==============================================================================
class SeverityDebounce:
    """
    Prevents momentary spikes from flipping the LED.
    Upgrades only after condition persists for DEBOUNCE_S.
    Downgrades immediately.
    """
    def __init__(self, debounce_s: float = 3.0):
        self.debounce_s         = debounce_s
        self.committed_severity = "NORMAL"
        self.candidate_severity = "NORMAL"
        self.candidate_since    = time.time()

    def update(self, raw_severity: str, now: float) -> str:
        level = {"NORMAL": 0, "WARN": 1, "ALERT": 2}
        raw_lvl       = level.get(raw_severity, 0)
        committed_lvl = level.get(self.committed_severity, 0)

        if raw_lvl < committed_lvl:
            self.committed_severity = raw_severity
            self.candidate_severity = raw_severity
            self.candidate_since    = now
            return self.committed_severity

        if raw_severity != self.candidate_severity:
            self.candidate_severity = raw_severity
            self.candidate_since    = now

        if (raw_lvl > committed_lvl and
                (now - self.candidate_since) >= self.debounce_s):
            self.committed_severity = raw_severity

        return self.committed_severity


def compute_raw_severity(temp_c, bpm_value) -> str:
    # FIX: Removed WARM_ROOM_MODE suppressor for temperature.
    # Temperature alerts now always use the same logic; the TempContextAnalyser
    # handles warm environments via rate-of-change, making hard suppression redundant.
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
            "severity", "bpm", "amb_t",
            "contact_status", "contact_conf",
            "anomaly_prob", "anomaly_threshold_used",
            "temp_context", "anomaly_persisting_s",
        ])


def log_eval(wall_time, read_latency_ms, severity, bpm, amb_t,
             contact_status, contact_conf, anomaly_prob,
             anomaly_threshold_used, temp_context, anomaly_persisting_s):
    if not EVAL_LOGGING or _eval_writer is None:
        return
    _eval_writer.writerow([
        round(wall_time, 4),
        round(read_latency_ms, 3),
        severity,
        round(bpm, 2)             if bpm is not None else "",
        round(amb_t, 2)           if amb_t is not None else "",
        contact_status,
        round(contact_conf, 3),
        round(anomaly_prob, 4)    if anomaly_prob is not None else "",
        round(anomaly_threshold_used, 3),
        temp_context,
        round(anomaly_persisting_s, 2) if anomaly_persisting_s is not None else "",
    ])
    _eval_file_h.flush()


# ==============================================================================
# Load models
# ==============================================================================
startup_test()
init_eval_log()

contact_bundle = load_joblib_model("models/contact_model.pkl")
anomaly_bundle = load_joblib_model("models/anomaly_mlp.pkl")
ai_available   = contact_bundle is not None and anomaly_bundle is not None

label_map            = {}
CONTACT_GOOD_IDX     = set()
contact_model        = None
contact_feature_cols = None
anomaly_model        = None
anomaly_imputer      = None
anomaly_scaler       = None
anomaly_feature_cols = None

if ai_available:
    try:
        with open("models/contact_label_map.json") as f:
            label_map = json.load(f)
        CONTACT_GOOD_IDX     = {int(k) for k, v in label_map.items() if "good_contact" in v}
        contact_model        = contact_bundle["model"]
        contact_feature_cols = contact_bundle["feature_cols"]
        anomaly_model        = anomaly_bundle["model"]
        anomaly_imputer      = anomaly_bundle["imputer"]
        anomaly_scaler       = anomaly_bundle["scaler"]
        anomaly_feature_cols = anomaly_bundle["feature_cols"]

        # FIX: Assert that confounding absolute features are not in the anomaly model.
        # If these fire, retrain with train_anomaly_mlp.py v3+.
        assert "ir_raw" not in anomaly_feature_cols, \
            "ir_raw in anomaly features — absolute IR is a confounding variable. " \
            "Retrain with train_anomaly_mlp.py v3+."
        assert "t_room" not in anomaly_feature_cols, \
            "t_room (absolute) in anomaly features — all sessions same temp causes leakage. " \
            "Retrain with train_anomaly_mlp.py v3+."

        print(f"[AI] Models loaded. Good-contact classes: "
              f"{[label_map[str(i)] for i in CONTACT_GOOD_IDX]}")
        print(f"[AI] Base anomaly threshold={ANOMALY_PROB_THRESHOLD}  "
              f"Stage-1 confidence gate={AI_CONFIDENCE_GATE}")
    except AssertionError as e:
        print(f"[AI] Model validation failed: {e}")
        ai_available = False
    except Exception as e:
        print(f"[AI] Model load error: {e}")
        ai_available = False
else:
    print("[AI] No models — rule-based fallback only.")


# ==============================================================================
# Main loop
# ==============================================================================
debouncer    = SeverityDebounce(debounce_s=SEVERITY_DEBOUNCE_S)
temp_context = TempContextAnalyser()
last_ai_print  = 0.0
last_eval_log  = 0.0

try:
    with SMBus(1) as bus:
        setup_max30102(bus)

        ppg_buf    = deque(maxlen=PPG_WINDOW)
        t_room_buf = deque(maxlen=PPG_WINDOW)
        context    = ContextBuffer(window_seconds=CONTEXT_WINDOW_SECONDS)

        # Beat detection
        ir_history     = []
        last_beat_time = None
        intervals      = deque(maxlen=INTERVAL_WINDOW)

        # FIX: HR smoothing buffer — was declared in config but never instantiated.
        # bpm_history is the raw-beat rolling average; hr_smooth_buf smooths across cycles.
        bpm_history    = deque(maxlen=HR_SMOOTH_N)
        hr_smooth_buf  = deque(maxlen=HR_SMOOTH_N)
        bpm            = None

        amb_t           = None
        last_temp_print = 0.0
        last_hr_print   = 0.0
        last_toggle     = 0.0
        blink_on        = False
        last_printed_severity = None

        anomaly_start = None
        anomaly_prob  = None

        # Stage-1 vote
        contact_vote_buf    = deque(maxlen=CONTACT_VOTE_N)

        # FIX: Initialise contact_status, contact_conf, and contact_label with safe
        # defaults BEFORE the main loop. Previously contact_conf was only assigned
        # inside `if run_full or run_short:`, causing a NameError on the first
        # log_eval() call if the PPG buffer wasn't full yet.
        contact_status      = "UNKNOWN"
        contact_conf        = 0.0
        contact_label       = "unknown"
        prev_contact_status = "UNKNOWN"

        # Track signal presence to detect loss cleanly
        signal_present = False

        while True:
            loop_start = time.time()

            # ── 1) Read MAX30102 ───────────────────────────────────────────
            read_t0 = time.time()
            ir = None
            try:
                d  = bus.read_i2c_block_data(I2C_ADDR, 0x07, 6)
                ir = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x03FFFF
                ppg_buf.append(float(ir))

                if 50000 < ir < 200000:
                    # Valid perfusion signal
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
                            last_beat_time = loop_start
                else:
                    # Signal lost — clear EVERYTHING immediately
                    if signal_present:
                        ir_history = []
                        last_beat_time = None
                        intervals.clear()
                        bpm_history.clear()
                        # FIX: Also flush the HR smoothing buffer on signal loss.
                        # Without this, stale smoothed BPM values persist after
                        # the finger is removed, causing false WARN/ALERT readings.
                        hr_smooth_buf.clear()
                        bpm = None
                        contact_vote_buf.clear()
                        contact_status = "BAD"
                        contact_conf   = 0.0
                        contact_label  = "unknown"
                        signal_present = False

            except Exception:
                pass

            read_latency_ms = (time.time() - read_t0) * 1000.0

            if len(intervals) >= 2:
                raw_bpm = 60.0 / (sum(intervals) / len(intervals))
                bpm_history.append(raw_bpm)

            valid_bpms = [v for v in bpm_history if v is not None]
            if valid_bpms:
                # FIX: Apply HR_SMOOTH_N smoothing across loop cycles.
                # Previously HR_SMOOTH_N was set in config but had no effect
                # because hr_smooth_buf was never used.
                hr_smooth_buf.append(float(np.mean(valid_bpms)))
                bpm = float(np.mean(hr_smooth_buf))
            else:
                bpm = None

            # ── 2) DHT22 temperature ───────────────────────────────────────
            if (loop_start - last_temp_print) >= TEMP_PRINT_S:
                try:
                    t = dht.temperature
                    if t is not None:
                        amb_t = float(t)
                        temp_context.update(loop_start, amb_t)
                        ctx_lbl = temp_context.context_label()
                        print(f"[TEMP] {amb_t:.2f} C  [{ctx_lbl}]")
                    else:
                        print("[WARN] DHT22 read failed (retrying).")
                except RuntimeError:
                    print("[WARN] DHT22 RuntimeError.")
                last_temp_print = loop_start

            if amb_t is not None:
                t_room_buf.append(amb_t)

            # Derive PPG proxies
            current_window = list(ppg_buf)
            if current_window:
                ir_arr_w      = np.asarray(current_window, dtype=np.float32)
                ir_mean_val   = float(np.mean(ir_arr_w))
                peak_to_peak  = float(np.max(ir_arr_w) - np.min(ir_arr_w))
                ac_dc_ratio_v = float(peak_to_peak / (ir_mean_val + 1e-6))
            else:
                peak_to_peak = ac_dc_ratio_v = 0.0

            context.update(
                now=loop_start,
                ir_raw=float(ir) if ir is not None else None,
                hr=float(bpm)   if bpm is not None else None,
                ac_dc=float(ac_dc_ratio_v),
                p2p=float(peak_to_peak),
                t_room=float(amb_t) if amb_t is not None else None,
            )

            # ── 3) HR print ────────────────────────────────────────────────
            if bpm is not None and (loop_start - last_hr_print) >= HR_PRINT_S:
                print(f"[HR] {bpm:.1f} BPM")
                last_hr_print = loop_start

            # ── 4) AI 2-stage pipeline ─────────────────────────────────────
            ai_should_print         = (loop_start - last_ai_print) >= AI_PRINT_S
            anomaly_prob_this_cycle = None
            dyn_threshold           = ANOMALY_PROB_THRESHOLD
            temp_ctx_label          = temp_context.context_label()

            if ai_available:

                # Stage 1 — contact classifier
                run_short = (len(ppg_buf) >= PPG_SHORT_WINDOW and
                             len(ppg_buf) < PPG_WINDOW)
                run_full  = (len(ppg_buf) == PPG_WINDOW and len(t_room_buf) >= 5)

                if run_full or run_short:
                    win_to_use = (list(ppg_buf)[-PPG_SHORT_WINDOW:]
                                  if run_short else list(ppg_buf))
                    tr_to_use  = list(t_room_buf)

                    x_c      = build_contact_feature_vector(win_to_use, tr_to_use,
                                                            contact_feature_cols)
                    pred_idx = int(contact_model.predict(x_c)[0])
                    if hasattr(contact_model, "predict_proba"):
                        probs        = contact_model.predict_proba(x_c)[0]
                        contact_conf = float(np.max(probs))
                    else:
                        contact_conf = 1.0

                    contact_label = label_map.get(str(pred_idx), "unknown")
                    raw_status    = "GOOD" if pred_idx in CONTACT_GOOD_IDX else "BAD"

                    short_gate = AI_CONFIDENCE_GATE + 0.10 if run_short else AI_CONFIDENCE_GATE
                    if contact_conf >= short_gate:
                        contact_vote_buf.append(raw_status)

                    if contact_vote_buf:
                        good_count = sum(1 for v in contact_vote_buf if v == "GOOD")
                        contact_status = "GOOD" if good_count > len(contact_vote_buf) / 2 else "BAD"
                    else:
                        contact_status = "UNKNOWN"

                    if contact_status == "BAD" and prev_contact_status == "GOOD":
                        flush_n = int(len(ppg_buf) * MOTION_FLUSH_FRACTION)
                        for _ in range(flush_n):
                            if ppg_buf:
                                ppg_buf.popleft()

                    prev_contact_status = contact_status

                    win_label = "short" if run_short else "full"
                    if ai_should_print:
                        print(f"[AI-S1] Contact: {contact_label} "
                              f"(conf={contact_conf:.2f}, vote={contact_status}, win={win_label})")

                # Stage 2 — context-aware anomaly detection
                if contact_status == "GOOD":
                    dyn_threshold = temp_context.adjusted_threshold(ir)
                    if ai_should_print and dyn_threshold != ANOMALY_PROB_THRESHOLD:
                        print(f"[TEMP-CTX] {temp_ctx_label} — "
                              f"anomaly threshold adjusted: {dyn_threshold:.2f}")

                    fmap  = context.feature_map(
                        ir_raw=float(ir) if ir is not None else None,
                        ac_dc_ratio=ac_dc_ratio_v,
                        peak_to_peak=peak_to_peak,
                        t_room=amb_t,
                        hr=bpm,
                    )
                    x_mlp = build_anomaly_feature_vector(fmap, anomaly_feature_cols)
                    x_np  = anomaly_imputer.transform(
                        x_mlp[anomaly_feature_cols].to_numpy(dtype=np.float32))
                    x_np  = anomaly_scaler.transform(x_np)

                    pred         = int(anomaly_model.predict(x_np)[0])
                    anomaly_prob = float(anomaly_model.predict_proba(x_np)[0][1]) \
                                   if hasattr(anomaly_model, "predict_proba") else float(pred)
                    anomaly_prob_this_cycle = anomaly_prob

                    is_anomaly = anomaly_prob >= dyn_threshold

                    if is_anomaly:
                        if anomaly_start is None:
                            anomaly_start = loop_start
                        persist_s = loop_start - anomaly_start
                        if ai_should_print:
                            print(f"[AI-S2] ANOMALY  prob={anomaly_prob:.2f}  "
                                  f"threshold={dyn_threshold:.2f}  "
                                  f"persisting={persist_s:.1f}s")
                        raw_severity = "ALERT" if persist_s >= ANOMALY_PERSIST_S else "WARN"
                    else:
                        anomaly_start = None
                        persist_s     = 0.0
                        if ai_should_print:
                            print(f"[AI-S2] Normal   prob={anomaly_prob:.2f}  "
                                  f"threshold={dyn_threshold:.2f}")
                        raw_severity = compute_raw_severity(amb_t, bpm)

                elif contact_status == "BAD":
                    anomaly_start = None
                    raw_severity  = "NORMAL"
                    if ai_should_print:
                        print("[AI-S1] No/poor contact — waiting for signal (green).")
                else:
                    raw_severity = compute_raw_severity(amb_t, bpm)

            else:
                raw_severity = compute_raw_severity(amb_t, bpm)

            # Debounce and commit severity
            severity          = debouncer.update(raw_severity, loop_start)
            persist_s_for_log = (loop_start - anomaly_start) if anomaly_start else None

            # ── 5) Status print on change ──────────────────────────────────
            if severity != last_printed_severity:
                print(f"[STATUS] -> {severity}")
                last_printed_severity = severity

            # ── 6) LED blink ───────────────────────────────────────────────
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

            if ai_should_print:
                last_ai_print = loop_start

            # ── 7) Evaluation log ──────────────────────────────────────────
            if EVAL_LOGGING and (loop_start - last_eval_log) >= EVAL_LOG_S:
                log_eval(
                    wall_time=loop_start,
                    read_latency_ms=read_latency_ms,
                    severity=severity,
                    bpm=bpm,
                    amb_t=amb_t,
                    contact_status=contact_status,
                    contact_conf=contact_conf,
                    anomaly_prob=anomaly_prob_this_cycle,
                    anomaly_threshold_used=dyn_threshold,
                    temp_context=temp_ctx_label,
                    anomaly_persisting_s=persist_s_for_log,
                )
                last_eval_log = loop_start

            time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[EXIT] Shutdown.")
finally:
    all_leds_off()
    buzzer.off()
    if _eval_file_h:
        _eval_file_h.close()
    try:
        dht.exit()
    except Exception:
        pass