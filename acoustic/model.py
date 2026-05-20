"""
acoustic/model.py — Task B
===========================
CNN architecture for MFCC-based cough classification.

Input:  (batch, 40, 128, 1)   — MFCC spectrogram
Output: (batch, 4)             — softmax probabilities
                                 [Normal, Asthma, Pneumonia, TB]
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from acoustic.preprocess import INPUT_SHAPE, NUM_CLASSES, CLASSES


def build_cough_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Build and compile the CNN model.

    Architecture:
        Block 1: Conv2D(32)  → BN → ReLU → MaxPool → Dropout
        Block 2: Conv2D(64)  → BN → ReLU → MaxPool → Dropout
        Block 3: Conv2D(128) → BN → ReLU → GlobalAvgPool
        Head:    Dense(256)  → Dropout(0.4) → Dense(4, softmax)

    Parameters
    ----------
    input_shape : tuple   e.g. (40, 128, 1)
    num_classes : int     4 (Normal, Asthma, Pneumonia, TB)

    Returns
    -------
    tf.keras.Model — compiled, ready to train
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required: pip install tensorflow")

    inputs = keras.Input(shape=input_shape, name='mfcc_input')

    # ── Block 1 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='drop1')(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='drop2')(x)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.4, name='drop_fc')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='CoughCNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    model = build_cough_cnn()
    model.summary()
    print(f"\nClasses: {CLASSES}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Output classes: {NUM_CLASSES}")
