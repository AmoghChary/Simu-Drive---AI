 SimuDrive-AI 🚗🧠

SimuDrive-AI is a Python-based self-driving car simulation project. It integrates multiple computer vision modules including object detection using YOLOv5, lane detection, and traffic sign recognition using deep learning. This project is built as a part of a research initiative and is modular, allowing easy expansion and testing.



## 🔧 Features

- 🚦 **Traffic Sign Recognition** using a trained CNN model
- 🛣️ **Lane Detection** using OpenCV and gradient masking
- 🧍 **Object Detection** with YOLOv5 (local repo)
- 📹 Video-based inference from simulated or recorded driving footage
- 📁 Modular code with clear separation of models and utilities

---

## 📁 Project Structure
SimuDrive-AI/
├── yolov5-master/ # YOLOv5 local repo
├── Models/
│ ├── Yolo.py # YOLO inference script
│ ├── train_traffic_sign_model.py # Training script for traffic sign classifier
│ └── traffic_sign_model.h5 # Trained CNN model for sign recognition
├── Utils/
│ ├── lane_detector.py # Lane detection logic
│ └── predict_sign.py # Prediction script using CNN
├── test_videos/
│ └── road1.mp4 # Sample input video
├── README.md
├── .gitignore

