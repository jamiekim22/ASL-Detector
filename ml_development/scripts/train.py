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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing import image

np.random.seed(5)
tf.random.set_seed(2)


def load_images(directory, labels):
    """
    Walks through `directory` ('./ml_development/data/asl_train', contains one subdirectory per class label)
    and reads all images, resizing the original 200x200 to 64x64 pixels.
    Returns a tuple (images_array, labels_array).
    """
    images = []
    y = []
    for idx, label in enumerate(labels):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            filepath = os.path.join(label_dir, fname)
            img = cv2.imread(filepath)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            images.append(img)
            y.append(idx)
    return np.array(images, dtype=np.float32), np.array(y, dtype=np.int32)


def print_images(image_list, labels, uniq_labels, title):
    """
    Plot one sample of each class from `image_list` in a grid.
    """
    n = len(uniq_labels)
    cols = 8
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, label in enumerate(uniq_labels):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
        plt.title(label)
        ax.axis('off')
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, uniq_labels, title):
    """
    Plot a confusion matrix for the true vs. predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title(title, fontsize=24)
    plt.colorbar()
    tick_marks = np.arange(len(uniq_labels))
    plt.xticks(tick_marks, uniq_labels, rotation=45, ha="right")
    plt.yticks(tick_marks, uniq_labels)
    plt.xlabel('Predicted label', fontsize=18)
    plt.ylabel('True label', fontsize=18)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.show()


def build_model(num_classes):
    """
    Creates and returns a Keras Sequential CNN model.
    """
    model = Sequential([
        Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(64, 64, 3)),
        Conv2D(64, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((4, 4)),
        Dropout(0.5),

        Conv2D(128, (5, 5), activation='relu', padding='same'),
        Conv2D(128, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((4, 4)),
        Dropout(0.5),

        Conv2D(256, (5, 5), activation='relu', padding='same'),
        Dropout(0.5),

        Flatten(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_dir = os.path.join(base_dir, "ml_development", "data", "asl_train")
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(save_dir, exist_ok=True)

    uniq_labels = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(uniq_labels)
    print(f"Discovered {num_classes} classes: {uniq_labels}")

    X, y = load_images(train_dir, uniq_labels)
    print(f"Loaded {X.shape[0]} training images.")

    # Splitting a test set out of the training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    print(f"Training on {X_train.shape[0]} images; testing on {X_test.shape[0]} images.")

    print_images(X_train[:num_classes], y_train, uniq_labels, title="Sample Training Images")
    print_images(X_test[:num_classes], y_test, uniq_labels, title="Sample Test Images")

    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=num_classes)

    # Normalize pixel RGB values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='loss', patience=2, cooldown=0),
        EarlyStopping(monitor='accuracy', patience=2, min_delta=1e-4)
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=10,
        batch_size=64,
        callbacks=callbacks
    )

    # Evaluation on test set
    test_score = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_score[1] * 100:.2f}%")

    model_save_path = os.path.join(save_dir, "asl_model.h5")
    model.save(model_save_path)
    print(f"Saved trained model to {model_save_path}")


    '''
    BELOW CONTAINS ALL THE PLOTS
    '''
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], label="Train Loss")
    plt.plot(epochs, history.history['accuracy'], label="Train Acc")
    plt.plot(epochs, history.history.get('val_loss', []), label="Val Loss")
    plt.plot(epochs, history.history.get('val_accuracy', []), label="Val Acc")
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    
    metrics_plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(metrics_plot_path)
    print(f"Saved training metrics plot to {metrics_plot_path}")
    plt.show()

    # Confusion matrix
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(y_test, y_test_pred, uniq_labels, "Confusion Matrix: Test Set")

    # Example single prediction
    sample_path = os.path.join(train_dir, "space", "space2000.jpg")
    if os.path.exists(sample_path):
        img = image.load_img(sample_path, target_size=(64, 64))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        predicted_label = uniq_labels[np.argmax(preds)]
        print(f"Predicted label for {sample_path}: {predicted_label}")


if __name__ == "__main__":
    main()