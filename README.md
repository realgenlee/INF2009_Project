
# AI-Enhanced Edge-Based Remote Patient Monitoring System

⚠️ **Project Status: In Development**

This repository contains the implementation of a **privacy-preserving edge-based remote patient monitoring system** running on a **Raspberry Pi 5**.

The system integrates physiological sensors, edge-based machine learning, and a local alert system to detect abnormal patient conditions in real time.

Initial development used **threshold-based rules** to validate the sensing pipeline. The system has now been extended with **machine learning models running locally on the Raspberry Pi**.

---

# Project Overview

Remote patient monitoring systems often rely on cloud processing to analyse physiological data. This introduces several challenges:

- network dependency
- higher latency
- potential privacy risks
- increased bandwidth usage

This project explores an **edge computing approach**, where physiological signals are analysed directly on the Raspberry Pi.

By performing analytics at the edge, the system can:

- reduce latency
- preserve patient privacy
- operate without continuous internet connectivity
- trigger immediate local alerts

---

# System Architecture

The system follows a **multi-layer edge architecture**:

Sensors → Feature Extraction → AI Models → Alert System

### Sensing Layer

The Raspberry Pi collects data from multiple sources:

| Sensor | Purpose |
|------|------|
| MAX30102 | PPG signal for heart rate detection |
| DHT22 | Ambient temperature monitoring |

These sensors provide **physiological and environmental context** for anomaly detection.

---

### Edge Analytics Layer

All analytics are performed **directly on the Raspberry Pi**.

The system uses a **two-stage AI pipeline** to improve reliability and reduce false alarms.

#### Stage 1 — Contact Quality Classification

A machine learning classifier determines whether the sensor reading represents **valid physiological contact**.

Possible classes include:

- good_contact
- poor_contact
- motion_artifact
- finger_off

This prevents invalid signals from triggering health alerts.

---

#### Stage 2 — Anomaly Detection

If good contact is detected, the system evaluates physiological patterns to determine whether the readings are abnormal.

Features used for anomaly detection include:

- AC/DC ratio of the PPG signal
- peak-to-peak amplitude
- signal stability
- heart rate variability
- temperature change

A lightweight **MLP classifier** is used for anomaly detection.

---

### Alert Layer

When an abnormal condition persists, the system triggers a local alert using LEDs and a buzzer.

| Status | Indicator |
|------|------|
| Normal | Green LED |
| Warning | Yellow LED |
| Critical | Red LED + Buzzer |

A **persistence gate** ensures anomalies must persist for several seconds before an alert is raised.

---

# Project Workflow

The project pipeline consists of four main stages.

## 1. Data Collection

Training data is collected using:

data_logger.py

This script reads sensor data and stores labelled samples for different contact conditions.

Example:

python data_logger.py --label good_contact --duration 120  
python data_logger.py --label finger_off --duration 90  
python data_logger.py --label poor_contact --duration 90  
python data_logger.py --label motion_artifact --duration 90

The collected data is stored in:

training_data/

---

## 2. Model Training

Two machine learning models are trained from the collected data.

### Contact Classifier

train_contact_classifier.py

This model classifies signal quality.

### Anomaly Detection Model

train_anomaly_mlp.py

This model detects abnormal physiological patterns.

Both models are exported to the **models/** directory for deployment.

---

## 3. Edge Deployment

The monitoring system runs on the Raspberry Pi using:

python main_ai.py

At runtime the system:

1. reads sensor signals  
2. extracts features  
3. evaluates contact quality  
4. performs anomaly detection  
5. triggers alerts if necessary  

All processing occurs **locally on the Raspberry Pi**.

---

# Hardware Components

| Component | Purpose |
|------|------|
| Raspberry Pi 5 | Edge computing device |
| MAX30102 | Heart rate and PPG sensor |
| DHT22 | Temperature sensor |
| LEDs | Status indicators |
| Buzzer | Alert notification |

---

# Edge AI Design Considerations

The system was designed with edge deployment constraints in mind:

- lightweight machine learning models
- minimal latency
- low power consumption
- minimal cloud dependency

Initial rule-based thresholds were used to validate the sensing pipeline before transitioning to machine learning models.

---

# Evaluation Plan

The system will be evaluated using the following metrics:

| Metric | Description |
|------|------|
| Detection latency | Time from sensor reading to alert |
| False positive rate | Alerts triggered during normal conditions |
| Model accuracy | Classification accuracy on validation data |
| System resilience | Operation during network disconnection |

---

# Future Work

Planned improvements include:

- dashboard for visualising sensor data and AI predictions
- expanded training dataset
- model optimisation for edge inference
- additional physiological sensors
- improved anomaly detection models

---

# Repository Structure

.
├── data_logger.py  
├── train_contact_classifier.py  
├── train_anomaly_mlp.py  
├── main_ai.py  
├── training_data/  
└── models/

---

# Project Goals

This project aims to demonstrate that **edge-based AI systems can provide reliable real-time monitoring while preserving privacy and reducing dependence on cloud infrastructure.**
