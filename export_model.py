import tensorflow as tf
import os

MODEL_PATH = 'best_model.keras'
TFLITE_PATH = 'plant_classifier.tflite'

def export_to_tflite():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # FP16 Quantization
    print("Applying Float16 Quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Model exported to {TFLITE_PATH}")
    print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export_to_tflite()
