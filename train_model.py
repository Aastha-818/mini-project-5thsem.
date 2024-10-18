# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import shutil
from sklearn.model_selection import train_test_split

class DentalModelTrainer:
    def __init__(self):
        """Initialize the Dental Disease Model Trainer"""
        self.base_dir = 'organized_dataset'
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.valid_dir = os.path.join(self.base_dir, 'valid')
        self.test_dir = os.path.join(self.base_dir, 'test')
        self.disease_classes = [
            'Calculus',
            'Caries_Gingivitus_ToothDiscoloration',
            'Data_caries',
            'Gingivitis',
            'hypodontia',
            'Mouth_Ulcer',
            'Tooth_Discoloration'
        ]
        self.input_shape = (224, 224, 3)
        self.model = None

    def organize_dataset(self, source_dir='old/archive (3)'):
        """Organize dataset into train, validation, and test sets"""
        print(f"Organizing dataset from {source_dir}")
        
        # Create directories
        for dir_path in [self.base_dir, self.train_dir, self.valid_dir, self.test_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        
        # Create class subdirectories
        for class_name in self.disease_classes:
            for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
                os.makedirs(os.path.join(dir_path, class_name))

        # Split and copy files
        for class_name in self.disease_classes:
            source_class_dir = os.path.join(source_dir, class_name)
            if not os.path.exists(source_class_dir):
                print(f"Warning: Directory not found - {source_class_dir}")
                continue

            images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                print(f"Warning: No images found in {source_class_dir}")
                continue

            # Split into train (70%), validation (15%), and test (15%)
            train_images, temp = train_test_split(images, train_size=0.7, random_state=42)
            valid_images, test_images = train_test_split(temp, train_size=0.5, random_state=42)

            # Copy files
            for img_list, dest_dir in [
                (train_images, self.train_dir),
                (valid_images, self.valid_dir),
                (test_images, self.test_dir)
            ]:
                for img in img_list:
                    shutil.copy2(
                        os.path.join(source_class_dir, img),
                        os.path.join(dest_dir, class_name, img)
                    )

    def build_model(self):
        """Build and compile the CNN model"""
        print("Building model...")
        
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.disease_classes), activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, epochs=20, batch_size=32):
        """Train the model"""
        print("Training model...")

        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        valid_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)
        ]

        # Train
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=valid_generator,
            callbacks=callbacks
        )

        return history

    def save_model(self, model_path='dental_model.h5'):
        """Save the trained model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

def main():
    """Main function to train the model"""
    try:
        trainer = DentalModelTrainer()
        
        # Organize dataset
        trainer.organize_dataset()
        
        # Build and train model
        trainer.build_model()
        history = trainer.train_model()
        
        # Save model
        trainer.save_model()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()