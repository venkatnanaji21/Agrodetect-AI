import tensorflow as tf
import numpy as np
import json
import os

class PlantPredictor:
    def __init__(self, model_path='best_model.keras', class_names_path='class_names.txt'):
        self.output_type = 'keras'
        if model_path.endswith('.tflite'):
            self.output_type = 'tflite'
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)
            
        self.class_names = self._load_class_names(class_names_path)
        
    def _load_class_names(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Class names file not found at {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # MobileNetV2 preprocess (if using keras model, it might be included in the model if we used the layer, 
        # but standard is to map inputs to [-1, 1])
        # In our train_model.py we included tf.keras.applications.mobilenet_v2.preprocess_input inside the model.
        # But for TFLite or raw logic, we should check. 
        # Since we included `preprocess_input` layer in the Keras model, we don't need to manually preprocess for Keras model.
        # However, for TFLite, the model input expects whatever the Keras model input expects.
        return img_array

    def predict(self, image_path, top_k=3):
        img_array = self.preprocess_image(image_path)
        
        if self.output_type == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            # Keras model includes preprocessing layer
            predictions = self.model.predict(img_array, verbose=0)[0]
            
        # Get top k
        top_indices = predictions.argsort()[-top_k:][::-1]
        results = []
        for i in top_indices:
            results.append({
                "class": self.class_names[i],
                "confidence": float(predictions[i])
            })
            
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict plant disease from image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="best_model.keras", help="Path to model file (.keras or .tflite)")
    args = parser.parse_args()
    
    try:
        predictor = PlantPredictor(model_path=args.model)
        results = predictor.predict(args.image)
        
        print(f"\nPredictions for {args.image}:")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['class']}: {res['confidence']:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
