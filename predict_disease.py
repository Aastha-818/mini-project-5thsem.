# predict_disease.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

class UnknownDiseaseError(Exception):
    """Custom exception for unknown dental diseases"""
    pass

class DentalDiseasePredictor:
    def __init__(self, model_path='dental_model.h5', confidence_threshold=0.3):
        """
        Initialize the predictor with the trained model
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Minimum confidence threshold to accept a prediction
                                If max confidence is below this, assume unknown disease
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224, 3)
        self.confidence_threshold = confidence_threshold
        self.disease_classes = [
            'Caries',
            'Gingivitis',
            'Hypodontia',
            'Mouth Ulcer',
            'Tooth_discoloration'
        ]

    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = load_model(self.model_path)

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Preprocess
        img = cv2.resize(img, self.input_shape[:2])
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    def predict(self, image_path):
        """
        Predict dental disease from image using the trained model
        Raises UnknownDiseaseError if the image likely contains an unknown disease
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Preprocess image
            img = self.preprocess_image(image_path)

            # Make prediction
            predictions = self.model.predict(img)[0]
            max_confidence = float(np.max(predictions))
            
            # Check if the maximum confidence is below threshold
            if max_confidence < self.confidence_threshold:
                all_probabilities = {
                    disease: float(prob) * 100 
                    for disease, prob in zip(self.disease_classes, predictions)
                }
                raise UnknownDiseaseError(
                    f"Detected possible unknown disease. Confidence too low for known diseases.\n"
                    f"Highest confidence was {max_confidence * 100:.2f}% "
                    f"(threshold: {self.confidence_threshold * 100:.2f}%)\n"
                    f"Probabilities for known diseases:\n"
                    + "\n".join(f"- {disease}: {prob:.2f}%" 
                               for disease, prob in all_probabilities.items())
                )

            predicted_class_index = np.argmax(predictions)

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': self.disease_classes[idx],
                    'confidence': float(predictions[idx]) * 100
                }
                for idx in top_3_indices
            ]

            return {
                'predicted_disease': self.disease_classes[predicted_class_index],
                'confidence': max_confidence * 100,
                'top_3_predictions': top_3_predictions,
                'all_probabilities': {
                    disease: float(prob) * 100 
                    for disease, prob in zip(self.disease_classes, predictions)
                }
            }

        except UnknownDiseaseError:
            raise
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main function to demonstrate prediction"""
    image_path = 'old\hypodontia.jpeg'  # Replace with your image path

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        predictor = DentalDiseasePredictor()
        result = predictor.predict(image_path)

        if 'error' in result:
            print(f"Error: {result['error']}")
            return

        print("\nPrediction Results:")
        print(f"Predicted Disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        
        print("\nTop 3 Predictions:")
        for pred in result['top_3_predictions']:
            print(f"{pred['disease']}: {pred['confidence']:.2f}%")
        
        print("\nAll Disease Probabilities:")
        for disease, prob in result['all_probabilities'].items():
            print(f"{disease}: {prob:.2f}%")

    except UnknownDiseaseError as e:
        print(f"\nUnknown Disease Detected:\n{str(e)}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
