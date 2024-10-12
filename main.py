import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

from flask import Flask, request, jsonify
import os

class DentalDiseaseDetectionSystem:
    def __init__(self):
        # Initialize both models
        self.binary_model = self.build_binary_model()
        self.multiclass_model = self.build_multiclass_model()
        self.disease_classes = ['Cavity', 'Gingivitis', 'Periodontitis', 'Dental Calculus', 'Dental Abscess']
        
    def build_binary_model(self):
        """Build binary classification model (Disease/No Disease)"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Dense Layers
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_multiclass_model(self):
        """Build multi-class classification model (5 specific diseases)"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(5, activation='softmax')  # 5 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess the input image"""
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=0)
            
        except Exception as e:
            raise Exception(f"Error in preprocessing image: {str(e)}")
    
    def train_binary_model(self, train_data_dir, validation_data_dir, epochs=20):
        """Train the binary classification model"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',  # binary classification
            classes=['normal', 'disease']
        )
        
        validation_generator = valid_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            classes=['normal', 'disease']
        )
        
        history = self.binary_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // 32,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // 32
        )
        
        return history
    
    def train_multiclass_model(self, train_data_dir, validation_data_dir, epochs=20):
        """Train the multi-class classification model"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',  # multi-class classification
            classes=self.disease_classes
        )
        
        validation_generator = valid_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            classes=self.disease_classes
        )
        
        history = self.multiclass_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // 32,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // 32
        )
        
        return history
    
    def predict(self, image_path):
        """Two-stage prediction process"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Stage 1: Binary Classification
            binary_prediction = self.binary_model.predict(processed_image)[0][0]
            
            # If no disease is detected, return early
            if binary_prediction < 0.5:
                return {
                    'has_disease': False,
                    'message': 'No dental disease detected',
                    'confidence': float(1 - binary_prediction)
                }
            
            # Stage 2: Multi-class Classification
            multiclass_prediction = self.multiclass_model.predict(processed_image)[0]
            predicted_class = np.argmax(multiclass_prediction)
            confidence = float(multiclass_prediction[predicted_class])
            
            # Check confidence threshold
            if confidence < 0.5:
                return {
                    'has_disease': True,
                    'message': 'Unknown dental condition detected. Please consult a dentist.',
                    'confidence': float(binary_prediction)
                }
            
            # Return specific disease prediction
            return {
                'has_disease': True,
                'disease_type': self.disease_classes[predicted_class],
                'confidence': confidence,
                'message': f'Detected {self.disease_classes[predicted_class]}'
            }
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")
    
    def save_models(self, binary_path='binary_model.h5', multiclass_path='multiclass_model.h5'):
        """Save both models"""
        self.binary_model.save(binary_path)
        self.multiclass_model.save(multiclass_path)
    
    def load_models(self, binary_path='binary_model.h5', multiclass_path='multiclass_model.h5'):
        """Load both models"""
        self.binary_model = load_model(binary_path)
        self.multiclass_model = load_model(multiclass_path)

# Flask Web Application
app = Flask(__name__)
detection_system = DentalDiseaseDetectionSystem()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the path of the local image
        image_file = './old/image.jpg'
        
        # Check if the file exists
        if not os.path.exists(image_file):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Use the detection system to predict
        result = detection_system.predict(image_file)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load trained models if they exist
    if os.path.exists('binary_model.h5') and os.path.exists('multiclass_model.h5'):
        detection_system.load_models()
    app.run(debug=True)