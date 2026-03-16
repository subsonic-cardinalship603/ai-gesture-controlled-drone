import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# --- 1. Initialize TFLite Model ---
try:
    interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Auto-detect required model channels (1 for Grayscale, 3 for RGB)
    # input_shape format: [1, 96, 96, channels]
    input_shape = input_details[0]['shape']
    MODEL_CHANNELS = input_shape[3] 
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

IMG_SIZE = 96
CONFIDENCE_THRESHOLD = 0.4

# Gesture label mapping based on TensorFlow's alphabetical folder sorting
# (0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9)
gesture_labels = {
    0: "Folder_0",   # Index 0  -> Folder 0
    1: "Folder_1",   # Index 1  -> Folder 1
    2: "Folder_10",  # Index 2  -> Folder 10
    3: "Folder_11",  # Index 3  -> Folder 11
    4: "Folder_12",  # Index 4  -> Folder 12
    5: "Folder_13",  # Index 5  -> Folder 13
    6: "Folder_14",  # Index 6  -> Folder 14
    7: "Folder_15",  # Index 7  -> Folder 15
    8: "Folder_16",  # Index 8  -> Folder 16
    9: "Folder_17",  # Index 9  -> Folder 17
    10: "Folder_18", # Index 10 -> Folder 18
    11: "Folder_19", # Index 11 -> Folder 19
    12: "Folder_2",  # Index 12 -> Folder 2
    13: "Folder_3",  # Index 13 -> Folder 3
    14: "Folder_4",  # Index 14 -> Folder 4
    15: "Folder_5",  # Index 15 -> Folder 5
    16: "Folder_6",  # Index 16 -> Folder 6
    17: "Folder_7",  # Index 17 -> Folder 7
    18: "Folder_8",  # Index 18 -> Folder 8
    19: "Folder_9"   # Index 19 -> Folder 9
}

# --- 2. Initialize MediaPipe Hand Landmarker ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. Start Camera ---
cap = cv2.VideoCapture(0)

print("Starting Gesture Recognition... Check 'Model Input' window: Hand should be WHITE.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Mirror the frame and convert to RGB for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Perform hand detection
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in hand_landmarks]
            y_list = [int(lm.y * h) for lm in hand_landmarks]

            # Calculate bounding box with padding
            padding = 40 
            x_min, x_max = max(0, min(x_list)-padding), min(w, max(x_list)+padding)
            y_min, y_max = max(0, min(y_list)-padding), min(h, max(y_list)+padding)

            # Crop the hand region
            hand_img_bgr = frame[y_min:y_max, x_min:x_max]
            
            if hand_img_bgr.size > 0:
                # --- Image Preprocessing (Binary Conversion) ---
                # 1. Grayscale
                gray = cv2.cvtColor(hand_img_bgr, cv2.COLOR_BGR2GRAY)
                # 2. Gaussian Blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # 3. Binary Thresholding (Hand=White, Background=Black)
                # Use THRESH_BINARY_INV if background is brighter than the hand
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   

                # 4. Resize to match model input
                resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
                
                # Debug window: Shows what the CNN actually sees
                cv2.imshow("Model Input (Binary)", resized)

                # 5. Dimension adjustment based on model requirements
                if MODEL_CHANNELS == 1:
                    # Input shape: (1, 96, 96, 1)
                    input_data = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
                else:
                    # Input shape: (1, 96, 96, 3)
                    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    input_data = np.expand_dims(resized_rgb, axis=0).astype(np.float32)

                # NOTE: Normalized inside the model if train_model.py included layers.Rescaling(1./255)
                # No manual division by 255.0 here to avoid double-scaling

                # --- Inference ---
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                # Get result with highest probability
                gesture_id = np.argmax(prediction)
                confidence = np.max(prediction)

                # Display Logic
                if confidence > CONFIDENCE_THRESHOLD:
                    gesture_name = gesture_labels.get(gesture_id, "Unknown")
                    color = (0, 255, 0) # Green for valid detection
                else:
                    gesture_name = "Uncertain"
                    color = (0, 255, 255) # Yellow for low confidence

                # Draw bounding box and text on main frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, f"{gesture_name} {confidence:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the main frame
    cv2.imshow("AI Gesture Controller", frame)
    
    # Exit on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27: break

# Resource Release
cap.release()
cv2.destroyAllWindows()
