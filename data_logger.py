import argparse
import csv
import os
import time
from collections import deque
from typing import Optional, Tuple

import adafruit_dht
import board
import numpy as np
from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory
from smbus2 import SMBus

Device.pin_factory = LGPIOFactory()

# -- Sensor config --------------------------------------------------------------
I2C_ADDR = 0x57
DHT22_PIN = board.D4
SAMPLE_HZ = 50                  # target loop rate (Hz)
PPG_WINDOW = 250                # samples per contact window (5 s @ 50 Hz)
CONTEXT_WINDOW_SECONDS = 5.0    # rolling context features
OUTPUT_DIR = "training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Heart-rate estimation config
MIN_BPM = 40
MAX_BPM = 200
INTERVAL_WINDOW = 6

REFRACTORY_S = 0.45       # minimum time between beats
BEAT_BUF_LEN = 25         # short IR history for beat detection
MIN_PROMINENCE = 150.0    # minimum dip depth
FINGER_PRESENT_IR = 1000  # accepts both ~1200 and ~120k style readings

# SpO2 estimation config
SPO2_MIN = 70.0
SPO2_MAX = 100.0

# Expected CSV header used for schema validation on append.
EXPECTED_HEADER = [
    "timestamp",
    "red_raw",
    "ir_raw",
    "ac_dc_ratio",
    "peak_to_peak",
    "t_room",
    "hr",
    "spo2_est",
    "hr_std_5s",
    "ir_std_5s",
    "ir_slope",
    "ac_dc_ratio_std",
    "peak_to_peak_std",
    "t_room_std_5s",
    "dt_room",
    "hr_valid",
    "ir_drop",
    "low_variation",
    "p2p_cv",
    "label",
]


# -- MAX30102 helpers -----------------------------------------------------------
def setup_max30102(bus: SMBus) -> None:
    bus.write_byte_data(I2C_ADDR, 0x09, 0x40)   # Reset
    time.sleep(0.1)
    bus.write_byte_data(I2C_ADDR, 0x09, 0x03)   # Mode: Red + IR
    bus.write_byte_data(I2C_ADDR, 0x0C, 0x1F)   # Red LED current
    bus.write_byte_data(I2C_ADDR, 0x0D, 0x1F)   # IR LED current
    bus.write_byte_data(I2C_ADDR, 0x0A, 0x27)   # Pulse width


def read_red_ir(bus: SMBus) -> Tuple[Optional[int], Optional[int]]:
    try:
        d = bus.read_i2c_block_data(I2C_ADDR, 0x07, 6)
        red = ((d[0] << 16) | (d[1] << 8) | d[2]) & 0x03FFFF
        ir = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x03FFFF
        return red, ir
    except Exception:
        return None, None


# -- Feature helpers ------------------------------------------------------------
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


def ac_dc_ratio(window) -> float:
    vals = [float(v) for v in window if v is not None and np.isfinite(v)]
    if not vals:
        return 0.0
    dc = float(np.mean(vals))
    ac = float(np.max(vals) - np.min(vals))
    return ac / (dc + 1e-6)


def peak_to_peak(window) -> float:
    vals = [float(v) for v in window if v is not None and np.isfinite(v)]
    return float(np.max(vals) - np.min(vals)) if vals else 0.0


def estimate_spo2(red_window, ir_window) -> Optional[float]:
    red_vals = np.array(
        [float(v) for v in red_window if v is not None and np.isfinite(v)],
        dtype=float
    )
    ir_vals = np.array(
        [float(v) for v in ir_window if v is not None and np.isfinite(v)],
        dtype=float
    )

    if len(red_vals) < 10 or len(ir_vals) < 10:
        return None

    dc_red = float(np.mean(red_vals))
    dc_ir = float(np.mean(ir_vals))
    ac_red = float(np.max(red_vals) - np.min(red_vals))
    ac_ir = float(np.max(ir_vals) - np.min(ir_vals))

    if dc_red <= 0 or dc_ir <= 0 or ac_ir <= 0:
        return None

    ratio_r = (ac_red / (dc_red + 1e-6)) / (ac_ir / (dc_ir + 1e-6))

    # Simple prototype approximation, not medical-grade calibration
    spo2 = 110.0 - 25.0 * ratio_r
    spo2 = max(SPO2_MIN, min(SPO2_MAX, spo2))
    return float(spo2)


class ContextBuffer:
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = float(window_seconds)
        self.time_buf = deque()
        self.ir_buf = deque()
        self.hr_buf = deque()
        self.acdc_buf = deque()
        self.p2p_buf = deque()
        self.t_room_buf = deque()

    def update(self, now: float, ir_raw: Optional[float], hr: Optional[float],
               ac_dc: Optional[float], p2p: Optional[float], t_room: Optional[float]) -> None:
        self.time_buf.append(now)
        self.ir_buf.append(ir_raw)
        self.hr_buf.append(hr)
        self.acdc_buf.append(ac_dc)
        self.p2p_buf.append(p2p)
        self.t_room_buf.append(t_room)
        self._prune(now)

    def _prune(self, now: float) -> None:
        while self.time_buf and (now - self.time_buf[0]) > self.window_seconds:
            self.time_buf.popleft()
            self.ir_buf.popleft()
            self.hr_buf.popleft()
            self.acdc_buf.popleft()
            self.p2p_buf.popleft()
            self.t_room_buf.popleft()

    def features(self) -> dict:
        t_room_vals = [v for v in self.t_room_buf if v is not None and np.isfinite(v)]
        dt_room = 0.0
        if len(t_room_vals) >= 2:
            dt_room = float(t_room_vals[-1] - t_room_vals[-2])

        p2p_mean = safe_mean(self.p2p_buf)
        ir_mean = safe_mean(self.ir_buf)

        return {
            "hr_std_5s": safe_std(self.hr_buf),
            "ir_std_5s": safe_std(self.ir_buf),
            "ir_slope": safe_slope(self.time_buf, self.ir_buf),
            "ac_dc_ratio_std": safe_std(self.acdc_buf),
            "peak_to_peak_std": safe_std(self.p2p_buf),
            "t_room_std_5s": safe_std(self.t_room_buf),
            "dt_room": dt_room,
            "hr_valid": 1.0 if any(v is not None and np.isfinite(v) for v in self.hr_buf) else 0.0,
            "ir_drop": 1.0 if ir_mean > 0.0 and ir_mean < 5000.0 else 0.0,
            "low_variation": 1.0 if p2p_mean < 1000.0 else 0.0,
            "p2p_cv": float(p2p_mean / (ir_mean + 1e-6)) if ir_mean > 0.0 else 0.0,
        }

# -- Main -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        choices=[
            "good_contact",
            "finger_off",
            "poor_contact",
            "motion_artifact",
            "cold_temp",
        ],
    )
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    args = parser.parse_args()

    label = args.label
    duration = args.duration
    outfile = os.path.join(OUTPUT_DIR, f"{label}.csv")
    is_new = not os.path.exists(outfile)

    if not is_new:
        with open(outfile, "r", newline="") as check_f:
            existing_header = check_f.readline().strip().split(",")
        if existing_header != EXPECTED_HEADER:
            raise RuntimeError(
                f"Schema mismatch in '{outfile}'.\n"
                f"  Existing : {existing_header}\n"
                f"  Expected : {EXPECTED_HEADER}\n"
                f"Delete the file and re-collect, or rename it before recording."
            )

    dht = adafruit_dht.DHT22(DHT22_PIN, use_pulseio=False)

    print(f"[LOG] Recording '{label}' for {duration}s -> {outfile}")
    print("[LOG] Press Ctrl+C to stop early.\n")

    ir_ppg_buf = deque(maxlen=PPG_WINDOW)
    red_ppg_buf = deque(maxlen=PPG_WINDOW)
    context = ContextBuffer(window_seconds=CONTEXT_WINDOW_SECONDS)
    t_room = None
    rows = 0

    # HR estimation state
    beat_buf = deque(maxlen=BEAT_BUF_LEN)
    last_beat_time = None
    intervals = deque(maxlen=INTERVAL_WINDOW)
    bpm = None

    with SMBus(1) as bus, open(outfile, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(EXPECTED_HEADER)

        setup_max30102(bus)
        end_time = time.time() + duration
        last_dht = 0.0
        period = 1.0 / SAMPLE_HZ

        try:
            while time.time() < end_time:
                loop_start = time.time()

                if (loop_start - last_dht) >= 2.0:
                    try:
                        t = dht.temperature
                        if t is not None:
                            t_room = float(t)
                    except RuntimeError:
                        pass
                    last_dht = loop_start

                red, ir = read_red_ir(bus)

                if ir is not None:
                    ir_ppg_buf.append(float(ir))
                if red is not None:
                    red_ppg_buf.append(float(red))

                if ir is not None:
                    # HR uses IR channel
                    if ir > FINGER_PRESENT_IR:
                        beat_buf.append(float(ir))

                        if len(beat_buf) >= 3:
                            prev_v = beat_buf[-3]
                            curr_v = beat_buf[-2]
                            next_v = beat_buf[-1]

                            mean_v = float(np.mean(beat_buf))
                            std_v = float(np.std(beat_buf))
                            prominence = max(MIN_PROMINENCE, 0.8 * std_v)

                            is_local_min = (curr_v < prev_v) and (curr_v < next_v)
                            deep_enough = (mean_v - curr_v) > prominence
                            far_enough = (last_beat_time is None) or ((loop_start - last_beat_time) > REFRACTORY_S)

                            if is_local_min and deep_enough and far_enough:
                                if last_beat_time is not None:
                                    interval = loop_start - last_beat_time
                                    inst_bpm = 60.0 / interval if interval > 0 else None
                                    if inst_bpm is not None and (MIN_BPM <= inst_bpm <= MAX_BPM):
                                        intervals.append(interval)
                                        bpm = 60.0 / (sum(intervals) / len(intervals))
                                last_beat_time = loop_start
                    else:
                        beat_buf.clear()
                        last_beat_time = None
                        intervals.clear()
                        bpm = None

                if len(intervals) >= 2:
                    bpm = 60.0 / (sum(intervals) / len(intervals))

                ir_window = list(ir_ppg_buf)
                red_window = list(red_ppg_buf)

                adc_r = ac_dc_ratio(ir_window)
                p2p = peak_to_peak(ir_window)

                # Only estimate SpO2 when both channels have enough data and a finger is likely present
                spo2_est = None
                if ir is not None and red is not None and ir > FINGER_PRESENT_IR:
                    spo2_est = estimate_spo2(red_window, ir_window)

                context.update(
                    now=loop_start,
                    ir_raw=float(ir) if ir is not None else None,
                    hr=float(bpm) if bpm is not None else None,
                    ac_dc=float(adc_r),
                    p2p=float(p2p),
                    t_room=float(t_room) if t_room is not None else None,
                )
                ctx = context.features()

                writer.writerow([
                    round(loop_start, 4),
                    red if red is not None else "",
                    ir if ir is not None else "",
                    round(adc_r, 6),
                    round(p2p, 2),
                    round(t_room, 2) if t_room is not None else "",
                    round(bpm, 2) if bpm is not None else "",
                    round(spo2_est, 2) if spo2_est is not None else "",
                    round(ctx["hr_std_5s"], 6),
                    round(ctx["ir_std_5s"], 6),
                    round(ctx["ir_slope"], 6),
                    round(ctx["ac_dc_ratio_std"], 6),
                    round(ctx["peak_to_peak_std"], 6),
                    round(ctx["t_room_std_5s"], 6),
                    round(ctx["dt_room"], 6),
                    round(ctx["hr_valid"], 1),
                    round(ctx["ir_drop"], 1),
                    round(ctx["low_variation"], 1),
                    round(ctx["p2p_cv"], 6),
                    label,
                ])
                rows += 1

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, period - elapsed))

        except KeyboardInterrupt:
            print("\n[LOG] Stopped early.")

    print(f"[LOG] Saved {rows} rows to {outfile}")
    try:
        dht.exit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
