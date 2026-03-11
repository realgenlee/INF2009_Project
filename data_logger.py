import argparse
import csv
import os
import time
from collections import deque
from typing import Optional

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

# Expected CSV header — used for schema validation on append.
# If an existing file's header doesn't match this list exactly, the script
# will refuse to append rather than silently produce misaligned rows.
EXPECTED_HEADER = [
    "timestamp",
    "ir_raw",
    "ac_dc_ratio",
    "peak_to_peak",
    "t_room",
    "hr",
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


def read_ir(bus: SMBus) -> Optional[int]:
    try:
        d = bus.read_i2c_block_data(I2C_ADDR, 0x07, 6)
        ir = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x03FFFF
        return ir
    except Exception:
        return None


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
        # FIX: Simplified to canonical label names only.
        # Legacy variants (good_contact_normal_room, good_contact_cold_room)
        # have been removed from the CLI to match what train_contact_classifier.py
        # expects after normalisation. This prevents confusion where the user
        # records under a legacy name and the file is silently remapped.
        choices=[
            "good_contact",
            "finger_off",
            "poor_contact",
            "motion_artifact",
        ],
    )
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    args = parser.parse_args()

    label = args.label
    duration = args.duration
    outfile = os.path.join(OUTPUT_DIR, f"{label}.csv")
    is_new = not os.path.exists(outfile)

    # FIX: Schema validation on append.
    # If an existing CSV was written with an older column layout, appending new
    # rows would silently misalign data (e.g. a value for 'ir_raw' landing in
    # the 'ac_dc_ratio' column). This check catches that before any data is lost.
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

    ppg_buf = deque(maxlen=PPG_WINDOW)
    context = ContextBuffer(window_seconds=CONTEXT_WINDOW_SECONDS)
    t_room = None
    rows = 0

    # HR estimation state
    ir_history = []
    last_beat_time = None
    intervals = deque(maxlen=INTERVAL_WINDOW)
    bpm = None

    with SMBus(1) as bus, open(outfile, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(EXPECTED_HEADER)

        setup_max30102(bus)
        end_time = time.time() + duration
        # NOTE: DHT22 is polled at most once every 2 seconds (per sensor spec).
        # This means most CSV rows will have t_room="" (empty/NaN). This is
        # intentional — pandas read_csv will convert empty strings to NaN, and
        # the imputer fills them correctly during training. Do NOT poll faster;
        # the DHT22 will return stale or error values if read more frequently.
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

                ir = read_ir(bus)
                if ir is not None:
                    ppg_buf.append(float(ir))

                    # same simple HR logic used in runtime
                    if 50000 < ir < 200000:
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
                        ir_history = []
                        last_beat_time = None
                        intervals.clear()
                        bpm = None

                if len(intervals) >= 2:
                    bpm = 60.0 / (sum(intervals) / len(intervals))

                window = list(ppg_buf)
                adc_r = ac_dc_ratio(window)
                p2p = peak_to_peak(window)

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
                    ir if ir is not None else "",
                    round(adc_r, 6),
                    round(p2p, 2),
                    round(t_room, 2) if t_room is not None else "",
                    round(bpm, 2) if bpm is not None else "",
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