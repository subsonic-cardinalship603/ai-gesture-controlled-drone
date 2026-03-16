# AI-Gesture-Controlled-Drone

This project develops an AI-based hand gesture recognition system designed to control a drone through real-time hand gestures. By leveraging MediaPipe for hand tracking and a custom CNN (Convolutional Neural Network) for classification, the system translates physical movements into drone commands (e.g., takeoff, land, movement) without the need for traditional controllers.

---

## Project Overview

The system features a hybrid pipeline that combines precise hand landmark detection with specialized binary image classification to ensure robust performance across various lighting conditions and backgrounds.

### Key Features:
- **Dual-Model Pipeline**: Uses MediaPipe for hand localization and a custom TFLite model for gesture classification.
- **Binary Processing**: Converts hand crops to black-and-white (binary) to focus on morphology rather than skin tone.
- **Apple Silicon Optimized**: Performance-tuned for Apple M5 chips using XNNPACK delegates.

---

## Model Performance
The model was trained for 20 epochs using a CNN architecture:
- **Training Accuracy**: 98.46%
- **Validation Accuracy**: 99.30%
- **Input Format**: 96x96 Grayscale Binary Image

---

## Dataset

The model is trained on a specialized version of the Hand Gesture Recognition Dataset.

- **Source**: [Hand Gesture Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset)
- **Total Images**: 24,000 (18,000 Train / 6,000 Test)
- **Classes**: 20 distinct gesture categories (0-19)
- **Input Specs**: 96x96 pixels, Grayscale (Single Channel)

---

## System Pipeline

The following workflow describes the real-time inference process:

```
Camera Input (RGB)
↓
MediaPipe Hand Landmarker (Hand Localization)
↓
Crop & Preprocess (Grayscale + Otsu's Thresholding)
↓
Custom CNN Model (96x96x1 TFLite)
↓
Gesture Classification (20 Classes)
↓
Drone Command Mapping
```

### Gesture Command Mapping (Example)


| Gesture (ID) | Command | Description |
| :--- | :--- | :--- |
| **OK (0)** | **Takeoff** | Start the motors and hover. |
| **Fist (11)** | **Land** | Secure landing at current position. |
| **Point (10)** | **Forward** | Move the drone forward. |
| **Rock (17)** | **Flip** | Perform a 360 degree stunt flip. |

---

## Technologies Used

- **Python 3.11**: Core development language.
- **MediaPipe**: For high-fidelity hand landmark detection.
- **TensorFlow / Keras**: Used for training the CNN classifier.
- **TensorFlow Lite**: For lightweight, real-time edge inference.
- **OpenCV**: For advanced image preprocessing and binary thresholding.
- **XNNPACK**: Optimized CPU inference for Apple M5.

---

## How to Use

### 1. Environment Setup
Activate the virtual environment and install dependencies:
```bash
source venv_detect/bin/activate
pip install mediapipe tensorflow opencv-python numpy
```
### 2. Required Models
Before running the detection, you must download the official MediaPipe model:
- **Hand Landmarker Bundle**: Download the `hand_landmarker.task` file from the [MediaPipe Official Models](https://ai.google.dev) page.
- **Placement**: Ensure `hand_landmarker.task` is placed in the project root directory.

### 3. Training the Model
To retrain the classifier using the grayscale binary approach:
```bash
python train_model.py
```

### 4. Running Real-Time Detection
Execute the main detection script:
```bash
python gesture_detection.py
```

---

## Future Improvements
- **Drone SDK Integration**: Connecting the command outputs to DJI Tello or ArduPilot.
- **3D Gesture Tracking**: Utilizing Z-axis data from MediaPipe for altitude control.
- **Robustness**: Adding more background-noise augmentation to the binary training set.

