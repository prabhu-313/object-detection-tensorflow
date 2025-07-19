import kaggle
# =============================================
# SYSTEM CHECKS AND ANNOUNCEMENTS
# =============================================
import os
import tensorflow as tf

print("✅ Python environment detected.")
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ GPU Available: {tf.config.list_physical_devices('GPU')}")
print("✅ Required folders: 'dataset/data/training_images' and 'train_solution_bounding_boxes.csv' should be in place.")
print("✅ Kaggle dataset must be downloaded and extracted prior to running this script if not already.")
print("✅ This code uses MobileNetV2 transfer learning for classification of object presence in car images.")
print("============================================================\n")
kaggle.api.authenticate()

kaggle.api.dataset_download_files('sshikamaru/car-object-detection', path='dataset/', unzip=True)

# Object Detection using TensorFlow with Custom Kaggle Dataset
# Internship Project 1

import kaggle
kaggle.api.authenticate()

# ## 1. Setup and Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from PIL import Image
import zipfile
import urllib.request
from sklearn.model_selection import train_test_split
import json

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ## 2. Kaggle Dataset Integration
# Downloading and preparing a custom dataset from Kaggle
# I'll use a popular object detection dataset



def setup_kaggle_dataset():
    """Download and prepare the Kaggle dataset"""
    import kaggle
    import os

    dataset_name = "sshikamaru/car-object-detection"
    download_path = "dataset/"

    os.makedirs(download_path, exist_ok=True)

    if not os.path.exists(os.path.join(download_path, 'data')):
        print(f"Downloading {dataset_name} from Kaggle...")
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Dataset already exists. Skipping download.")




class CustomDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path  # Should be 'dataset/data'
        self.class_names = ['car']
        self.num_classes = len(self.class_names)

    def load_dataset(self):
        """Load images and annotations from XLSX and folders"""
        import pandas as pd

        image_dir = os.path.join(self.dataset_path, 'training_images')
        print("Files in dataset/data:")
        print(os.listdir(self.dataset_path))
        bbox_file = os.path.join(self.dataset_path, 'train_solution_bounding_boxes.csv')

        print("Looking for Excel file at:", bbox_file)

        



        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(bbox_file):
            raise FileNotFoundError(f"Annotation file not found: {bbox_file}")

        # Read bounding box CSV
        df = pd.read_csv(bbox_file)
        


        images = []
        annotations = []

        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(image_dir, filename)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

                # Get annotations for this image
                records = df[df['image'] == filename]

                objects = []
                for _, row in records.iterrows():
                    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    objects.append({
                        'class': 'car',
                        'class_id': 0,
                        'bbox': bbox
                    })

                annotations.append({
                    'image_path': img_path,
                    'objects': objects
                })

        return np.array(images), annotations

    def preprocess_data(self, images, annotations):
        """Preprocess images and create training data"""
        # Normalize images
        images = images.astype(np.float32) / 255.0

        # Create labels for classification (binary: car present or not)
        labels = []
        for anno in annotations:
            label = [0] * self.num_classes
            for obj in anno['objects']:
                label[obj['class_id']] = 1
            labels.append(label)

        return images, np.array(labels)


# ## 4. Model Architecture
def create_object_detection_model(input_shape, num_classes):
    """Create a custom CNN model for object detection"""
    
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer for classification
        tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Multi-label classification
    ])
    
    return model

# ## 5. Transfer Learning with Pre-trained Model
def create_transfer_learning_model(num_classes):
    """Create model using transfer learning"""
    
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom classifier
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

# ## 6. Training Pipeline
def train_model(model, train_images, train_labels, val_images, val_labels, epochs=20):
    """Train the object detection model"""
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_images, train_labels,
        batch_size=16,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ## 7. Evaluation and Prediction
def evaluate_model(model, test_images, test_labels):
    """Evaluate model performance"""
    
    # Make predictions
    predictions = model.predict(test_images)
    
    # Calculate metrics
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_images, test_labels, verbose=0)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    return predictions

def visualize_predictions(images, true_labels, predictions, class_names, num_samples=5):
    """Visualize model predictions"""
    
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        
        # Get predicted and true classes
        pred_classes = [class_names[j] for j, val in enumerate(predictions[i]) if val > 0.5]
        true_classes = [class_names[j] for j, val in enumerate(true_labels[i]) if val == 1]
        
        plt.title(f'True: {true_classes}\nPred: {pred_classes}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ## 8. Main Execution Pipeline
def main_execution():
    """Main execution pipeline for the project"""
    
    print("="*60)
    print("    OBJECT DETECTION USING TENSORFLOW")
    print("    Custom Kaggle Dataset Implementation")
    print("="*60)
    
    # Step 1: Setup dataset
    print("\n1. Setting up Kaggle dataset...")
    setup_kaggle_dataset()

    # Step 2: Load dataset
    print("\n2. Loading dataset...")
    loader = CustomDatasetLoader('dataset/data')

    # loader = CustomDatasetLoader('dataset')
    images, annotations = loader.load_dataset()
    print(f"Loaded {len(images)} images")
    
    # Step 3: Preprocess data
    print("\n3. Preprocessing data...")
    processed_images, labels = loader.preprocess_data(images, annotations)
    
    # Resize images for model input
    processed_images = tf.image.resize(processed_images, [224, 224]).numpy()
    
    # Step 4: Split dataset
    print("\n4. Splitting dataset...")
    train_images, test_images, train_labels, test_labels = train_test_split(
        processed_images, labels, test_size=0.2, random_state=42
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Step 5: Create model
    print("\n5. Creating model...")
    model = create_transfer_learning_model(loader.num_classes)
    model.summary()
    
    # Step 6: Train model
    print("\n6. Training model...")
    history = train_model(model, train_images, train_labels, val_images, val_labels, epochs=10)
    
    # Step 7: Evaluate model
    print("\n7. Evaluating model...")
    predictions = evaluate_model(model, test_images, test_labels)
    
    # Step 8: Visualize results
    print("\n8. Visualizing results...")
    visualize_predictions(test_images, test_labels, predictions, loader.class_names)
    
    # Step 9: Plot training history
    plot_training_history(history)
    
    # Step 10: Save model
    model.save('models/final_object_detection_model.h5')
    print("\nModel saved successfully!")
    
    print("\n" + "="*60)
    print("    PROJECT EXECUTION COMPLETED!")
    print("="*60)

def plot_training_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# ## 9. Execute the Project
if __name__ == "__main__":
    main_execution()
