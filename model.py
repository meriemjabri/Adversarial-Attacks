import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Konstante für den Speicherpfad des Modells
MODEL_PATH = "mnist_resnet50_model.h5"

def load_or_train_model(x_train, y_train, x_test, y_test):
    """
    Trainiert oder lädt das ResNet50-Modell mit MNIST-Daten.

    Parameter:
        x_train, y_train: Vorverarbeitete Trainingsdaten und Labels.
        x_test, y_test: Vorverarbeitete Testdaten und Labels.

    Rückgabewert:
        tf.keras.Model: Das trainierte oder geladene Modell.
    """
    if not os.path.exists(MODEL_PATH):
        print("Modell nicht gefunden. Starte Training...")

        # Modellaufbau mit ResNet50 als Basis
        model = Sequential([
            ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(32, 32, 3)),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # ResNet50-Schichten einfrieren, um Training zu verhindern
        model.layers[0].trainable = False
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Datenaugmentation für das Training
        train_generator = ImageDataGenerator(rotation_range=40, shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
        val_generator = ImageDataGenerator()

        # Training- und Validierungs-Iteratoren
        train_iterator = train_generator.flow(x_train, y_train, batch_size=512, shuffle=True)
        val_iterator = val_generator.flow(x_test, y_test, batch_size=512, shuffle=False)

        # Modell trainieren
        model.fit(train_iterator, validation_data=val_iterator, epochs=5)

        # Trainiertes Modell speichern
        model.save(MODEL_PATH)
        print(f"Modell gespeichert unter {MODEL_PATH}")
    else:
        print("Trainiertes Modell gefunden. Lade gespeichertes Modell...")
        model = tf.keras.models.load_model(MODEL_PATH)
    
    return model
