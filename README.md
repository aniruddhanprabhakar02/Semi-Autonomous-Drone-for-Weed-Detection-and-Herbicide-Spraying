# Semi-Autonomous Drone for Weed Detection and Herbicide Spraying

## Overview

This project presents a semi-autonomous UAV-based precision agriculture system integrating real-time weed detection, edge AI inference, and targeted herbicide spraying. The system combines computer vision, embedded processing, and drone-based actuation to enable intelligent weed management with reduced chemical usage.

A lightweight YOLOv11n object detection model was optimized using Swin Transformer-based knowledge distillation and deployed on a Raspberry Pi 4B using TensorFlow Lite INT8 quantization for efficient real-time edge inference.

The system integrates:

* Real-time weed detection
* Raspberry Pi edge inference
* TensorFlow Lite optimization
* GPIO-based spraying control
* UAV-assisted targeted spraying
* Embedded AI deployment

---

# System Architecture

```text
USB Camera
     ↓
Raspberry Pi 4B
     ↓
YOLOv11n INT8 TFLite Inference
     ↓
Crop / Weed Detection
     ↓
Decision Logic
     ↓
GPIO Trigger
     ↓
MOSFET Driver Circuit
     ↓
Ultrasonic Spraying Mechanism
```

---

# Key Features

* Real-time crop and weed detection
* YOLOv11n lightweight object detection
* Swin Transformer-based knowledge distillation
* INT8 TensorFlow Lite optimization
* Raspberry Pi 4B deployment
* OpenCV-based image acquisition
* GPIO-based spraying control
* UAV-assisted precision spraying
* Edge AI inference pipeline
* Embedded computer vision system

---

# Hardware Components

## UAV Platform

* F450 Quadcopter Frame
* APM 2.8 Flight Controller
* NEO-7M GPS Module
* FLYSKY FS-i6 Transmitter & Receiver
* 2212 920KV BLDC Motors
* 30A ESCs
* 1045 Propellers
* 11.1V 2200mAh 3S LiPo Battery

## Embedded Computing

* Raspberry Pi 4B
* USB Camera

## Spraying System

* Ultrasonic Humidifier
* MOSFET Driver Circuit
* GPIO-Based Actuation

---

# Dataset Description

The weed detection dataset was prepared using the Roboflow platform.

## Dataset Information

* Total Images: 1300
* Classes:

  * crop
  * weed
* Annotation Format: YOLOv8
* Train/Validation/Test Split:

  * 70 / 20 / 10

## Preprocessing

* Auto-orientation
* EXIF removal
* Resize to 640×640
* Normalization

## Data Augmentation

* Horizontal flip
* Random rotation
* Brightness variation

---

# YOLOv11n Weed Detection Model

The weed detection pipeline uses a lightweight YOLOv11n object detection model optimized for embedded edge deployment.

## Model Features

* Single-stage object detector
* Real-time inference
* Lightweight architecture
* Multi-scale feature extraction
* Edge deployment optimized

---

# Model Evolution

The weed detection pipeline was developed in two stages.

## Stage 1 — Baseline YOLO Training

* Initial lightweight YOLO model training
* TensorFlow Lite export
* Raspberry Pi deployment validation

## Stage 2 — Knowledge Distillation Optimization

* Swin Transformer Tiny used as teacher model
* YOLOv11n used as lightweight student model
* Feature-based knowledge distillation using MSE loss
* INT8 TensorFlow Lite quantization for edge deployment

The final distilled INT8 model demonstrated improved lightweight inference suitability for real-time weed detection on Raspberry Pi 4B.

---

# Knowledge Distillation-Based Optimization

To improve lightweight edge inference performance, the YOLOv11n weed detection model was trained using a Swin Transformer-based knowledge distillation framework.

## Teacher Model

* Swin Transformer Tiny

## Student Model

* YOLOv11n

## Distillation Strategy

* Intermediate feature alignment
* Feature-based knowledge distillation
* Mean Squared Error (MSE) feature matching loss
* Combined YOLO detection loss + distillation loss

## Edge Deployment Optimization

The trained model was exported into multiple TensorFlow Lite formats:

* Float16 TFLite
* INT8 Quantized TFLite

The INT8 quantized model was deployed for efficient real-time inference on Raspberry Pi 4B.

---

# Quantization and Edge Deployment

The trained YOLOv11n model was optimized using TensorFlow Lite INT8 quantization to reduce:

* model size
* memory usage
* inference latency

## Deployment Target

* Raspberry Pi 4B

## Runtime

* TensorFlow Lite Runtime

## Inference Performance

* ~70–80 ms/image
* Real-time feasible edge inference

---

# Raspberry Pi GPIO-Based Spraying Control

A Raspberry Pi GPIO-based relay control mechanism was implemented for targeted herbicide spraying.

## Control Logic

```python
IF weed detected:
    Activate spraying system
ELSE:
    No action
```

## Spraying Pipeline

* GPIO HIGH → MOSFET ON → Sprayer ON
* GPIO LOW → MOSFET OFF → Sprayer OFF

The spraying mechanism uses an ultrasonic humidifier-based system for fine-mist herbicide application.

---

# Repository Structure

```text

Semi-Autonomous-Drone-for-Weed-Detection-and-Herbicide-Spraying/
│
├── Classes/
│   └── weed_classes.txt
│
├── code/
│   ├── Herbicide Spraying/
│   │   └── raspberrypi_spray_control.py
│   │
│   └── Weed Detection/
│       ├── yolov11n_realtime_inference.py
│       └── tflite_edge_inference.py
│
├── Crop_weedDataset.v2i.yolov8/
│
├── Demo Video/
│
├── Diagrams/
│
├── Images/
│
├── Models/
│   ├── yolov_distill_best_int8.tflite
│   └── yolov_distill_best_float16.tflite
│
├── Notebook/
│   ├── baseline_yolo_training.ipynb
│   └── swin_transformer_kd_training.ipynb
│
├── Training file/
│
├── .gitignore
├── README.md
└── Requirement.txt

---

# Installation

Clone the repository:

```bash
git clone https://github.com/aniruddhanprabhakar02/Semi-Autonomous-Drone-for-Weed-Detection-and-Herbicide-Spraying.git
cd Semi-Autonomous-Drone-for-Weed-Detection-and-Herbicide-Spraying
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Real-Time Weed Detection

## YOLOv11n Real-Time Detection

```bash
python inference/yolov11n_realtime_inference.py
```

## TFLite Edge Inference

```bash
python inference/tflite_edge_inference.py
```

---

# Performance Metrics

| Metric          | Value           |
| --------------- | --------------- |
| mAP@0.5         | 0.5852          |
| Precision       | 0.6822          |
| Recall          | 0.5778          |
| F1-Score        | 0.6257          |
| Inference Speed | ~70–80 ms/image |

The model demonstrates reliable real-time weed detection performance suitable for embedded agricultural applications.

---

# Applications

* Precision Agriculture
* Smart Farming
* UAV-Assisted Agriculture
* Embedded AI Systems
* Edge AI Deployment
* Computer Vision for Agriculture
* Intelligent Herbicide Spraying

---

# Limitations

* Dataset imbalance affects crop recall
* Environmental lighting variations impact detection
* UAV payload constraints
* Real-time actuation latency during motion

Future work may include:

* autonomous navigation
* improved dataset diversity
* multi-spectral sensing
* advanced edge optimization

---

# Intellectual Property Notice

Certain advanced architectures, integrated UAV control strategies, and closed-loop precision agriculture mechanisms associated with this work are part of ongoing research and protected system-level developments.

This repository is intended for academic, research, and portfolio demonstration purposes only.

---

# Author

ANIRUDDHAN P
M.E Embedded Systems and Technologies

---

# License

This repository is intended for educational and research demonstration purposes.
