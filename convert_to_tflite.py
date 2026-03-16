import tensorflow as tf

# load keras model
model = tf.keras.models.load_model("gesture_model.keras")

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save model
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as gesture_model.tflite")