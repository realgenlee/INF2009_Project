"""
dashboard_app.py
Subscribes to MQTT vitals topics and serves a live browser dashboard.
Run alongside main_ai_v4.py:
  python3 dashboard_app.py
Then open http://<pi-ip>:5000 in any browser on the same network.
"""

import json
import threading
import time

import paho.mqtt.client as mqtt
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BROKER_HOST   = "localhost"
BROKER_PORT   = 1883
FLASK_HOST    = "0.0.0.0"   # listen on all interfaces so other devices can connect
FLASK_PORT    = 5000
TOPICS        = ["vitals/hr", "vitals/contact", "vitals/anomaly", "vitals/env"]

# ---------------------------------------------------------------------------
# Shared state (written by MQTT thread, read by Flask/SocketIO)
# ---------------------------------------------------------------------------
_state = {
    "bpm":           None,
    "spo2":          None,
    "ir":            None,
    "red":           None,
    "contact_status": "UNKNOWN",
    "contact_label":  "unknown",
    "contact_conf":   0.0,
    "anomaly_score":  None,
    "anomaly_threshold": None,
    "is_anomaly":     False,
    "persisting_s":   None,
    "severity":       "WARN",
    "temp":           None,
    "temp_context":   "--",
    "last_update":    None,
}
_state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Flask + SocketIO
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "vitals-dashboard-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Dashboard HTML (single-file, no external dependencies except CDN)
# ---------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vitals Dashboard</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0f1117;
    --surface: #1a1d27;
    --card:    #20253a;
    --border:  #2e3450;
    --text:    #e2e5f0;
    --muted:   #7a80a0;
    --green:   #22c55e;
    --yellow:  #eab308;
    --red:     #ef4444;
    --blue:    #60a5fa;
    --teal:    #2dd4bf;
    --radius:  12px;
  }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, sans-serif;
    min-height: 100vh;
    padding: 1.5rem;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
  }
  header h1 { font-size: 1.1rem; font-weight: 500; letter-spacing: 0.02em; }
  #conn-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--muted); display: inline-block; margin-right: 6px;
    transition: background 0.4s;
  }
  #conn-dot.live { background: var(--green); }
  #last-update { font-size: 0.75rem; color: var(--muted); }

  /* Severity banner */
  #severity-banner {
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 1rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: background 0.4s, border-color 0.4s;
    border: 1px solid var(--border);
    background: var(--surface);
  }
  #severity-banner.NORMAL { background: #14291e; border-color: #22c55e55; }
  #severity-banner.WARN   { background: #2a2310; border-color: #eab30855; }
  #severity-banner.ALERT  { background: #2a1212; border-color: #ef444455; animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.75} }
  #severity-icon { font-size: 1.25rem; }

  /* Metric grid */
  .grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    margin-bottom: 1rem;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
  }
  .card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.4rem;
  }
  .card-value {
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
    transition: color 0.4s;
  }
  .card-unit  { font-size: 0.85rem; color: var(--muted); margin-left: 3px; }
  .card-sub   { font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }

  /* Contact status badge */
  #contact-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: background 0.4s, border-color 0.4s, color 0.4s;
    margin-top: 0.5rem;
  }
  #contact-badge.GOOD    { background:#14291e; border-color:#22c55e88; color:var(--green); }
  #contact-badge.BAD     { background:#2a2310; border-color:#eab30888; color:var(--yellow); }
  #contact-badge.UNKNOWN { background:var(--surface); color:var(--muted); }

  /* Anomaly score bar */
  .score-bar-wrap {
    margin-top: 0.6rem;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  #score-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--green);
    width: 50%;
    transition: width 0.5s, background 0.4s;
  }

  /* Mini history sparkline placeholder */
  #sparkline-wrap { margin-top: 1rem; }
  #sparkline-wrap canvas { width:100%; height:80px; }

  footer { margin-top: 2rem; font-size: 0.7rem; color: var(--muted); text-align: center; }
</style>
</head>
<body>
<header>
  <h1><span id="conn-dot"></span> Vitals Monitor</h1>
  <span id="last-update">Waiting for data…</span>
</header>

<!-- Severity banner -->
<div id="severity-banner" class="WARN">
  <span id="severity-icon">⬤</span>
  <span id="severity-text">Waiting for signal…</span>
</div>

<!-- Row 1: vital signs -->
<div class="grid">
  <div class="card">
    <div class="card-label">Heart rate</div>
    <div class="card-value" id="bpm-val">--<span class="card-unit">BPM</span></div>
    <div class="card-sub" id="bpm-sub">--</div>
  </div>
  <div class="card">
    <div class="card-label">SpO₂ (est.)</div>
    <div class="card-value" id="spo2-val">--<span class="card-unit">%</span></div>
    <div class="card-sub">Estimated</div>
  </div>
  <div class="card">
    <div class="card-label">IR signal</div>
    <div class="card-value" style="font-size:1.5rem" id="ir-val">--</div>
    <div class="card-sub" id="red-val">Red: --</div>
  </div>
  <div class="card">
    <div class="card-label">Temperature</div>
    <div class="card-value" id="temp-val">--<span class="card-unit">°C</span></div>
    <div class="card-sub" id="temp-ctx">--</div>
  </div>
</div>

<!-- Row 2: AI state -->
<div class="grid">
  <div class="card">
    <div class="card-label">Contact quality</div>
    <div class="card-value" style="font-size:1.1rem" id="contact-label-val">--</div>
    <span id="contact-badge" class="UNKNOWN">UNKNOWN</span>
    <div class="card-sub" id="contact-conf-val">Conf: --</div>
  </div>
  <div class="card">
    <div class="card-label">Anomaly score</div>
    <div class="card-value" style="font-size:1.1rem" id="score-val">--</div>
    <div class="score-bar-wrap"><div id="score-fill"></div></div>
    <div class="card-sub" id="score-sub">Threshold: --</div>
  </div>
  <div class="card">
    <div class="card-label">Anomaly persisting</div>
    <div class="card-value" style="font-size:1.5rem" id="persist-val">--<span class="card-unit">s</span></div>
    <div class="card-sub" id="anomaly-flag">--</div>
  </div>
</div>

<footer>Edge AI Vitals Monitor · MQTT → Flask · <span id="broker-info">localhost:1883</span></footer>

<script>
const socket = io();
const dot = document.getElementById('conn-dot');

socket.on('connect',    () => dot.classList.add('live'));
socket.on('disconnect', () => dot.classList.remove('live'));

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? '--';
}
function setHtml(id, val) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = val;
}

socket.on('state_update', d => {
  // Timestamp
  const now = new Date().toLocaleTimeString();
  setText('last-update', 'Updated ' + now);

  // Severity banner
  const sev = d.severity || 'WARN';
  const banner = document.getElementById('severity-banner');
  banner.className = sev;
  const icons = {NORMAL:'✅', WARN:'⚠️', ALERT:'🚨'};
  const msgs  = {
    NORMAL: 'Normal — all vitals within range',
    WARN:   'Warning — poor contact or signal unstable',
    ALERT:  'ALERT — anomaly detected!',
  };
  setText('severity-icon', icons[sev] || '⬤');
  setText('severity-text', msgs[sev] || sev);

  // Vitals
  const bpm = d.bpm != null ? d.bpm.toFixed(1) : '--';
  setHtml('bpm-val',  bpm + '<span class="card-unit">BPM</span>');
  const bpmEl = document.getElementById('bpm-val');
  if (bpmEl) bpmEl.style.color = d.bpm > 100 ? 'var(--yellow)' : d.bpm > 120 ? 'var(--red)' : 'var(--text)';

  const spo2 = d.spo2 != null ? d.spo2.toFixed(1) : '--';
  setHtml('spo2-val', spo2 + '<span class="card-unit">%</span>');
  const spo2El = document.getElementById('spo2-val');
  if (spo2El) spo2El.style.color = d.spo2 < 92 ? 'var(--red)' : d.spo2 < 95 ? 'var(--yellow)' : 'var(--teal)';

  setText('ir-val',  d.ir  != null ? d.ir.toLocaleString()  : '--');
  setText('red-val', d.red != null ? 'Red: ' + d.red.toLocaleString() : 'Red: --');
  setHtml('temp-val', (d.temp != null ? d.temp.toFixed(1) : '--') + '<span class="card-unit">°C</span>');
  setText('temp-ctx', d.temp_context || '--');

  // Contact
  const status = d.contact_status || 'UNKNOWN';
  setText('contact-label-val', d.contact_label || '--');
  const badge = document.getElementById('contact-badge');
  badge.textContent = status;
  badge.className = status;
  setText('contact-conf-val', d.contact_conf != null ? 'Conf: ' + (d.contact_conf * 100).toFixed(0) + '%' : 'Conf: --');

  // Anomaly score
  const score = d.anomaly_score;
  const thresh = d.anomaly_threshold;
  setText('score-val', score != null ? score.toFixed(4) : '--');
  setText('score-sub',  thresh != null ? 'Threshold: ' + thresh.toFixed(4) : 'Threshold: --');

  // Score bar — map score relative to threshold
  if (score != null && thresh != null) {
    const fill = document.getElementById('score-fill');
    // pct: score well above threshold = 100% green; score < threshold = 0% red
    const range = Math.abs(thresh) * 3 || 0.15;
    const pct = Math.max(0, Math.min(100, ((score - thresh) / range) * 100));
    fill.style.width = pct.toFixed(1) + '%';
    fill.style.background = d.is_anomaly ? 'var(--red)' : 'var(--green)';
  }

  // Persisting
  const ps = d.persisting_s;
  setHtml('persist-val', (ps != null ? ps.toFixed(1) : '--') + '<span class="card-unit">s</span>');
  const pflag = document.getElementById('anomaly-flag');
  if (pflag) {
    pflag.textContent = d.is_anomaly ? '⚠ Anomaly active' : 'Normal';
    pflag.style.color = d.is_anomaly ? 'var(--red)' : 'var(--muted)';
  }
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# MQTT subscriber
# ---------------------------------------------------------------------------
def _on_connect(client, userdata, flags, rc):
    if rc == 0:
        for topic in TOPICS:
            client.subscribe(topic, qos=0)
        print(f"[MQTT] Subscribed to {TOPICS}")
    else:
        print(f"[MQTT] Connection failed rc={rc}")


def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        return

    with _state_lock:
        topic = msg.topic
        if topic == "vitals/hr":
            _state.update({
                "bpm":   payload.get("bpm"),
                "spo2":  payload.get("spo2"),
                "ir":    payload.get("ir"),
                "red":   payload.get("red"),
            })
        elif topic == "vitals/contact":
            _state.update({
                "contact_status": payload.get("status", "UNKNOWN"),
                "contact_label":  payload.get("label", "unknown"),
                "contact_conf":   payload.get("conf", 0.0),
            })
        elif topic == "vitals/anomaly":
            _state.update({
                "anomaly_score":     payload.get("score"),
                "anomaly_threshold": payload.get("threshold"),
                "is_anomaly":        payload.get("is_anomaly", False),
                "persisting_s":      payload.get("persisting_s"),
                "severity":          payload.get("severity", "WARN"),
            })
        elif topic == "vitals/env":
            _state.update({
                "temp":         payload.get("temp"),
                "temp_context": payload.get("context", "--"),
            })
        _state["last_update"] = time.time()

    # Push to all connected browsers immediately
    with _state_lock:
        snapshot = dict(_state)
    socketio.emit("state_update", snapshot)


def _start_mqtt():
    client = mqtt.Client(client_id="dashboard_subscriber", clean_session=True)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    while True:
        try:
            client.connect(BROKER_HOST, BROKER_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"[MQTT] Subscriber error: {e} — retrying in 3s")
            time.sleep(3)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/state")
def state():
    with _state_lock:
        return dict(_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mqtt_thread = threading.Thread(target=_start_mqtt, daemon=True)
    mqtt_thread.start()
    print(f"[DASH] Dashboard running at http://0.0.0.0:{FLASK_PORT}")
    print(f"[DASH] Open http://<your-pi-ip>:{FLASK_PORT} in a browser")
    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, use_reloader=False)
