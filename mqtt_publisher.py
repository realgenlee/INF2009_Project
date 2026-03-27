"""
mqtt_publisher.py
Publishes sensor + AI state to MQTT topics.
Import this in main_ai_v4.py and call publish_vitals() once per cycle.

Topics:
  vitals/hr      -> {"bpm": 72.3, "spo2": 98.2, "ir": 131500, "red": 108900}
  vitals/contact -> {"status": "GOOD", "label": "good_contact", "conf": 0.94}
  vitals/anomaly -> {"score": 0.039, "threshold": -0.058, "is_anomaly": false, "persisting_s": 0}
  vitals/env     -> {"temp": 24.5, "context": "STABLE", "severity": "NORMAL"}
"""

import json
import threading
import time
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
KEEPALIVE   = 60
PUBLISH_HZ  = 10        # max publishes per second (matches DASHBOARD_LOG_S = 0.1)

_client      = None
_connected   = False
_last_pub    = 0.0
_pub_lock    = threading.Lock()


def _on_connect(client, userdata, flags, rc):
    global _connected
    if rc == 0:
        _connected = True
        print("[MQTT] Connected to broker.")
    else:
        _connected = False
        print(f"[MQTT] Connection failed rc={rc}")


def _on_disconnect(client, userdata, rc):
    global _connected
    _connected = False
    if rc != 0:
        print(f"[MQTT] Unexpected disconnect rc={rc} — will auto-reconnect.")


def init_mqtt():
    """Call once at startup, before the main loop."""
    global _client
    _client = mqtt.Client(client_id="pi_vitals_publisher", clean_session=True)
    _client.on_connect    = _on_connect
    _client.on_disconnect = _on_disconnect
    _client.reconnect_delay_set(min_delay=1, max_delay=10)
    try:
        _client.connect(BROKER_HOST, BROKER_PORT, KEEPALIVE)
        _client.loop_start()   # background thread — non-blocking
    except Exception as e:
        print(f"[MQTT] Could not connect: {e}")


def _pub(topic: str, payload: dict):
    """Internal: publish JSON payload if connected."""
    if _client is None or not _connected:
        return
    try:
        _client.publish(topic, json.dumps(payload), qos=0, retain=False)
    except Exception as e:
        print(f"[MQTT] Publish error on {topic}: {e}")


def publish_vitals(
    bpm, spo2_est, ir, red,
    contact_status, contact_label, contact_conf,
    anomaly_score, anomaly_threshold, is_anomaly, anomaly_persisting_s,
    amb_t, temp_context, severity,
):
    """
    Call once per main loop cycle.
    Rate-limited to PUBLISH_HZ — excess calls are silently dropped.
    All arguments accept None; None values are serialised as null in JSON.
    """
    global _last_pub
    now = time.monotonic()
    with _pub_lock:
        if (now - _last_pub) < (1.0 / PUBLISH_HZ):
            return
        _last_pub = now

    def _f(v, ndigits=2):
        return round(float(v), ndigits) if v is not None else None

    _pub("vitals/hr", {
        "bpm":   _f(bpm, 1),
        "spo2":  _f(spo2_est, 1),
        "ir":    int(ir)  if ir  is not None else None,
        "red":   int(red) if red is not None else None,
    })
    _pub("vitals/contact", {
        "status": contact_status,
        "label":  contact_label,
        "conf":   _f(contact_conf, 3),
    })
    _pub("vitals/anomaly", {
        "score":        _f(anomaly_score, 5),
        "threshold":    _f(anomaly_threshold, 5),
        "is_anomaly":   bool(is_anomaly),
        "persisting_s": _f(anomaly_persisting_s, 1),
        "severity":     severity,
    })
    _pub("vitals/env", {
        "temp":     _f(amb_t, 2),
        "context":  temp_context,
        "severity": severity,
    })


def stop_mqtt():
    """Call in the finally block of main_ai_v4.py."""
    global _client
    if _client:
        _client.loop_stop()
        _client.disconnect()
