# YOLO11 Snack Detection and Nutrition Analysis

## Project Overview

This project implements a real-time snack detection and nutrition analysis system using the YOLO11 object detection framework and OpenCV.

The system:

- Detects snack products in real-time using a webcam
- Uses a custom-trained YOLO11 model
- Displays bounding boxes and confidence scores
- Calculates total calories and protein for detected snacks
- Supports both image and webcam inference

---

# Features

- Custom YOLO11 object detection model
- Real-time webcam inference
- Nutrition overlay system
- GPU acceleration using NVIDIA CUDA
- OpenCV integration
- Label Studio dataset annotation
- Custom dataset training pipeline

---

# Snack Classes

The model was trained on the following classes:

- chips_ahoy
- chocolate_brownie
- chocolate_chip_cookie_dough
- lime
- nacho
- nilla
- nutter_butter
- oreo_mini
- ritz_cheese
- sea_salt

---

# Project Structure

```text
project_final/
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── images/
│   ├── labels/
│   ├── classes.txt
│   └── data.yaml
│
├── runs/
│   └── detect/
│       └── train-2/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── split_data.py
├── webcam_detect.py
├── webcam_nutrition.py
├── nutrition_data.yaml
└── README.md
```

---

# Environment Setup

## 1. Install Miniconda

Download and install Miniconda:

https://docs.conda.io/en/latest/miniconda.html

---

## 2. Create Python Environment

```bash
conda create -n yolo-env python=3.10 -y
conda activate yolo-env
```

---

# Install Dependencies

## Install Ultralytics YOLO

```bash
pip install ultralytics
```

## Install GPU-enabled PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Install OpenCV

```bash
pip install opencv-python
```

## Install Label Studio

```bash
pip install label-studio
```

## Install PyYAML

```bash
pip install pyyaml
```

---

# Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected Output:

```text
True
```

---

# Dataset Labeling

Launch Label Studio:

```bash
label-studio start
```

Label all snack images and export the dataset in YOLO format.

---

# Dataset Split

Run the dataset split script:

```bash
python split_data.py
```

This automatically creates:

- 80% training dataset
- 20% validation dataset

---

# data.yaml Configuration

Example:

```yaml
path: C:/Users/mjawe/project_final/data

train: train/images
val: val/images

nc: 10

names:
  - chips_ahoy
  - chocolate_brownie
  - chocolate_chip_cookie_dough
  - lime
  - nacho
  - nilla
  - nutter_butter
  - oreo_mini
  - ritz_cheese
  - sea_salt
```

---

# Training the Model

Train YOLO11 using:

```bash
yolo detect train model=yolo11s.pt data="data/data.yaml" epochs=60 imgsz=640 device=0
```

Training results are saved in:

```text
runs/detect/train-2/
```

Best model:

```text
runs/detect/train-2/weights/best.pt
```

---

# Testing on an Image

```bash
yolo detect predict model="runs/detect/train-2/weights/best.pt" source="C:/Users/mjawe/Desktop/test.jpg" conf=0.25 device=0
```

Prediction outputs are saved in:

```text
runs/detect/predict/
```

---

# Webcam Detection

Run real-time webcam inference:

```bash
python webcam_detect.py
```

Features:

- Live webcam detection
- Bounding boxes
- Confidence scores
- Real-time inference

Press:

```text
q
```

to quit the application.

---


# Nutrition Overlay System

Create a nutrition configuration file:

```text
nutrition_data.yaml
```

Example:

```yaml
classes:
  - name: oreo_mini
    nutrition:
      calories_per_serving: 140
      protein_grams: 1
```

Run:

```bash
python webcam_nutrition.py
```

The application displays:

- Detected snack labels
- Calories per serving
- Protein per serving
- Total calories detected
- Total protein detected
- Bounding boxes and confidence scores

---

# Model Performance

Final validation metrics:

| Metric | Value |
|---|---|
| Precision | 0.949 |
| Recall | 0.874 |
| mAP50 | 0.924 |
| mAP50-95 | 0.614 |

---

# Technologies Used

- YOLO11
- Ultralytics
- OpenCV
- PyTorch
- CUDA
- Python
- Label Studio

---

# Future Improvements

- Add more snack classes
- Improve dataset diversity
- Add calorie estimation by quantity
- Deploy on embedded systems
- Mobile application support
- Real-time nutritional recommendations

---

# Authors
Jawid Mohammadi
Julian Fong

