# Privacy-Preserving Edge-Based Remote Patient Vital Monitoring System

⚠️ **Project Status: In Development**

This repository currently contains the **system design, hardware configuration, and implementation plan** for an edge-based patient monitoring system.  
The **AI anomaly detection models, system evaluation, and experimental results are still under development** and will be added in future updates.

---

# Overview

This project implements a **privacy-preserving edge-based remote patient monitoring (RPM) system** using a **Raspberry Pi 5**.

The system collects physiological signals from sensors, processes the data locally on the edge device, detects abnormal conditions, and triggers **real-time alerts using LEDs and a buzzer**.

Unlike traditional monitoring systems that rely heavily on cloud infrastructure, this project adopts an **edge-first architecture** where critical data processing occurs locally.

Processing data directly on the edge device improves:

- **Latency** – faster detection of abnormal conditions  
- **Privacy** – sensitive patient data remains on the local device  
- **Reliability** – the system continues operating during network outages  

Edge computing has been identified as a promising approach for healthcare applications that require **low latency and secure handling of physiological data**.

---

# Problem Statement

Many existing remote patient monitoring systems rely on **cloud-centric architectures**, where raw physiological data is continuously transmitted to remote servers for analysis.

This approach introduces several challenges:

- **Latency** – delays in detecting abnormal health conditions
- **Network dependency** – monitoring fails during internet disruptions
- **Privacy risks** – sensitive patient data must traverse external networks
- **Bandwidth consumption** – constant data transmission increases network load

These issues become especially critical for **elderly patients or individuals with chronic health conditions**, where delayed alerts may impact patient safety.

There is therefore a need for a **monitoring system capable of detecting health anomalies locally while reducing reliance on continuous cloud connectivity**.

---

# Motivation and Importance

Remote patient monitoring systems are becoming increasingly important in modern healthcare, particularly in home-based care and telemedicine.

The global RPM market has been growing rapidly due to the increasing adoption of **telehealth and remote care technologies**.

However, concerns about:

- **data privacy**
- **system reliability**
- **network dependency**

remain major barriers to adoption.

Edge computing provides a potential solution by allowing physiological data to be processed **closer to where it is generated**, improving responsiveness and reducing exposure of sensitive health information.

This project explores how **edge computing can be used to improve reliability, privacy, and responsiveness in healthcare monitoring systems**.

---

# Project Goals

The primary goals of this project are:

1. Design and implement an edge-based remote patient monitoring system
2. Process physiological sensor data locally on a Raspberry Pi
3. Detect abnormal health conditions in real time
4. Trigger local alerts without relying on cloud connectivity
5. Reduce transmission of sensitive health data
6. Evaluate system performance in terms of latency, bandwidth usage, and reliability

These goals aim to demonstrate the feasibility of **edge-based healthcare monitoring systems**.

---

# System Architecture

The system follows a **three-layer architecture** consisting of a sensing layer, edge processing layer, and an optional cloud layer.

## 1. Sensing Layer

The sensing layer collects physiological signals from sensors connected to the Raspberry Pi.

Sensors used include:

- **MAX30102** – measures heart rate and blood oxygen saturation (SpO₂)
- **DHT22** – measures environmental temperature

These sensors generate continuous physiological data streams that are sent to the edge device for processing.

---

## 2. Edge Processing Layer (Raspberry Pi)

The **Raspberry Pi 5 acts as the edge computing device** responsible for:

- sensor data acquisition  
- signal preprocessing  
- feature extraction  
- anomaly detection  
- local alert generation  
- local data buffering  

When abnormal readings are detected, the Raspberry Pi triggers:

- LED indicators
- a buzzer alarm

This ensures that alerts can be generated **immediately without relying on external servers**.

The edge device therefore serves as the **core decision-making unit of the monitoring system**.

---

## 3. Cloud Layer (Optional)

The cloud layer is optional and is used primarily for:

- long-term storage of summarized health data
- visualization dashboards
- clinician monitoring

Only **summarized or non-sensitive data** is transmitted to the cloud to minimize privacy risks and reduce bandwidth usage.

---

# Hardware Setup

The system uses a Raspberry Pi as the edge computing platform connected to sensors and alert devices.

## Components

- Raspberry Pi 5
- MAX30102 Heart Rate Sensor
- DHT22 Temperature Sensor
- 3 Status LEDs (Green / Yellow / Red)
- Buzzer
- Resistors
- Jumper wires

---

# Wiring Diagram

![Hardware Wiring](docs/wiring_diagram.png)

---

## Sensor Connections

### DHT22 Temperature Sensor

| DHT22 Pin | Raspberry Pi |
|-----------|--------------|
| VCC | 5V |
| GND | GND |
| DATA | GPIO 4 |

The DHT22 sensor measures ambient temperature to provide environmental context for monitoring.

---

### MAX30102 Heart Rate Sensor

The MAX30102 communicates with the Raspberry Pi via **I²C**.

| MAX30102 Pin | Raspberry Pi |
|---------------|--------------|
| VIN | 3.3V |
| GND | GND |
| SDA | GPIO 2 |
| SCL | GPIO 3 |

This sensor measures:

- heart rate
- blood oxygen levels

---

## Alert System

The system includes a **local alert mechanism** using LEDs and a buzzer.

| Component | GPIO | Function |
|-----------|------|----------|
| Green LED | GPIO 17 | Normal status |
| Yellow LED | GPIO 27 | Warning |
| Red LED | GPIO 22 | Critical alert |
| Buzzer | GPIO 23 | Audio alarm |

Local alerts ensure that warnings are triggered **even when network connectivity is unavailable**.

---

# Design Considerations

Several approaches were considered for anomaly detection.

| Approach | Latency | Power Usage | Feasibility |
|----------|---------|-------------|-------------|
| Deep Learning Model | High | High | Not suitable |
| Machine Learning Classifier | Medium | Medium | Possible future work |
| Threshold-Based Detection | Low | Very Low | Selected |

Threshold-based logic was selected because it provides **low computational overhead and fast response time**, making it suitable for deployment on a Raspberry Pi.

---

# Evaluation Plan

The system will be evaluated based on:

- **Alert latency** (sensor reading → alert trigger)
- **Bandwidth usage reduction**
- **System resilience during network disruptions**
- **Power consumption of the edge device**

These metrics will be used to evaluate the effectiveness of the edge-based monitoring system.

---

# Expected Outcomes

The project aims to demonstrate that edge-based monitoring systems can:

- detect abnormal health conditions in real time
- reduce reliance on cloud infrastructure
- preserve patient privacy
- maintain monitoring functionality during connectivity failures
- reduce network bandwidth usage

The final output will be a **functional prototype of an edge-based remote patient monitoring system**.

---

# Future Work

Possible extensions include:

- machine learning-based anomaly detection
- predictive health analytics
- multi-patient monitoring systems
- secure cloud dashboards
- integration with healthcare platforms