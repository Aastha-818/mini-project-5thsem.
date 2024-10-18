# predict_disease.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

class DentalDiseasePredictor:
    def __init__(self, model_path='dental_model.h5'):
        """Initialize the predictor with a trained model"""
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224, 3)
        self.disease_classes = [
            'Calculus',
            'Caries_Gingivitus_ToothDiscoloration',
            'Data_caries',
            'Gingivitis',
            'hypodontia',
            'Mouth_Ulcer',
            'Tooth_Discoloration'
        ]

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = load_model(self.model_path)

    def predict(self, image_path):
        """Predict dental disease for a given image"""
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Preprocess
            img = cv2.resize(img, self.input_shape[:2])
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            predictions = self.model.predict(img)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': self.disease_classes[idx],
                    'confidence': float(predictions[0][idx]) * 100
                }
                for idx in top_3_indices
            ]

            return {
                'predicted_disease': self.disease_classes[predicted_class_index],
                'confidence': confidence * 100,
                'top_3_predictions': top_3_predictions
            }

        except Exception as e:
            return {'error': str(e)}

def main():
    """Main function to demonstrate prediction"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_disease.py <path_to_image>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        predictor = DentalDiseasePredictor()
        result = predictor.predict(image_path)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\nPrediction Results:")
            print(f"Predicted Disease: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print("\nTop 3 Predictions:")
            for pred in result['top_3_predictions']:
                print(f"{pred['disease']}: {pred['confidence']:.2f}%")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()