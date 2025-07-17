"""
Script for the ML model training:

1. Loads ASL alphabet images
2. Trains a convolutional neural network (CNN) on them
3. Evaluates its performance, and plots confusion matrices for both test and evaluation sets.
"""

import tensorflow as tf
import os
import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub as kgh
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical

def load_images(path):
    directory = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
    if os.path.exists(directory):
        uniq_labels = sorted([d for d in os.listdir(directory)
                              if os.path.isdir(os.path.join(directory, d))])
        num_classes = len(uniq_labels)
        print(f"Discovered {num_classes} classes: {uniq_labels}")

        X = []
        y = []
        for idx, label in enumerate(uniq_labels):
            label_dir = os.path.join(directory, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                filepath = os.path.join(label_dir, fname)
                img = cv2.imread(filepath)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))
                X.append(img)
                y.append(idx)
            print(f"Image loading {idx}/{len(uniq_labels)} completed.")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), uniq_labels
    else:
        raise FileNotFoundError(f"Directory {directory} not found.")

def split_data_by_label(X, y, uniq_labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train/validation/test sets for each label separately.
    
    Args:
        X: Input features (images)
        y: Labels
        uniq_labels: List of unique label names
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    print(f"Splitting data with ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    
    for label_idx, label_name in enumerate(uniq_labels):
        # Get indices for current label
        label_indices = np.where(y == label_idx)[0]
        num_samples = len(label_indices)
        
        if num_samples == 0:
            print(f"Warning: No samples found for label '{label_name}'")
            continue
        
        # Shuffle indices
        np.random.seed(random_state)
        np.random.shuffle(label_indices)
        
        # Calculate split points
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)
        
        # Split indices
        train_indices = label_indices[:train_end]
        val_indices = label_indices[train_end:val_end]
        test_indices = label_indices[val_end:]
        
        # Add to respective lists
        X_train.extend(X[train_indices])
        X_val.extend(X[val_indices])
        X_test.extend(X[test_indices])
        
        y_train.extend(y[train_indices])
        y_val.extend(y[val_indices])
        y_test.extend(y[test_indices])
        
        print(f"Label '{label_name}': {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    print(f"\nFinal split sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test, num_classes):
    """
    Preprocess the data for training.
    
    Args:
        X_train, X_val, X_test: Input features
        y_train, y_val, y_test: Labels
        num_classes: Number of classes
    
    Returns:
        Preprocessed datasets
    """
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"Data preprocessing completed:")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Number of classes: {num_classes}")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat

def build_cnn_model(input_shape, num_classes, config):
    """
    Build CNN model with specified configuration.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        config: Dictionary containing hyperparameters
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(config['filters'][0], (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(config['dropout']))
    
    # Additional convolutional blocks
    for i in range(1, len(config['filters'])):
        model.add(Conv2D(config['filters'][i], (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(config['dropout']))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(config['dense_units'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout']))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    else:
        optimizer = SGD(learning_rate=config['learning_rate'], momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Train the model with specified configuration.
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Training configuration
    
    Returns:
        Training history
    """
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['patience'] // 2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f"best_model_{config['name']}.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, y_test_cat, uniq_labels, config, history):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        y_test_cat: Categorical test labels
        uniq_labels: List of label names
        config: Model configuration
        history: Training history
    """
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=uniq_labels, yticklabels=uniq_labels)
    plt.title(f'Confusion Matrix - {config["name"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{config["name"]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {config["name"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {config["name"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{config["name"]}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    handle = "grassknoted/asl-alphabet"
    path = kgh.dataset_download(handle)

    X, y, uniq_labels = load_images(path)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_label(
        X, y, uniq_labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Preprocess data
    num_classes = len(uniq_labels)
    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat = preprocess_data(
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    )

    # Define different hyperparameter configurations
    configs = [
        {
            'name': 'config_1_basic',
            'filters': [32, 64, 128],
            'dense_units': 128,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'patience': 10,
            'optimizer': 'adam'
        },
        {
            'name': 'config_2_deep',
            'filters': [64, 128, 256, 512],
            'dense_units': 256,
            'dropout': 0.5,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': 10,
            'patience': 15,
            'optimizer': 'adam'
        },
        {
            'name': 'config_3_wide',
            'filters': [128, 256, 512],
            'dense_units': 512,
            'dropout': 0.4,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 10,
            'patience': 12,
            'optimizer': 'sgd'
        }
    ]

    # Train models with different configurations
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Training with configuration: {config['name']}")
        print(f"{'='*50}")
        
        # Build model
        model = build_cnn_model(X_train.shape[1:], num_classes, config)
        print(f"Model summary:")
        model.summary()
        
        # Train model
        history = train_model(model, X_train, y_train_cat, X_val, y_val_cat, config)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, y_test_cat, uniq_labels, config, history)
        
        # Save model
        model.save(f"asl_model_{config['name']}.h5")
        print(f"Model saved as asl_model_{config['name']}.h5")

if __name__ == "__main__":
    main()