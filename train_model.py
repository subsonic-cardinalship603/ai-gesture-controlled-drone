import tensorflow as tf
from tensorflow.keras import layers, models

# --- Configuration Parameters ---
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 20  # More epochs help stability since binary features are simpler

# --- 1. Load Dataset ---
# Set color_mode to 'grayscale' to match your black-and-white binary dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='grayscale' 
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='grayscale'
)

# Print detected classes (will be sorted alphabetically: 0, 1, 10, 11...)
class_names = train_dataset.class_names
print("Detected Classes:", class_names)

# --- 2. Build CNN Model ---
model = models.Sequential([
    # Input shape set to (96, 96, 1) for grayscale single-channel images
    layers.InputLayer(input_shape=(96, 96, 1)),
    
    # Data Augmentation: Helps model recognize gestures at different angles/sizes
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    # Normalization: Built-in rescaling to 0.0 - 1.0 range
    layers.Rescaling(1./255),

    # Convolutional Layers with Batch Normalization for faster convergence
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting (memorizing training images)
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. Start Training ---
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# --- 4. Save and Convert to TFLite ---
# Save the full Keras model
model.save("gesture_model.keras")

# Convert the Keras model to a lightweight TFLite model for real-time inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Write the .tflite file to the project directory
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Binary-based TFLite model saved successfully!")
