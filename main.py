# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from fgsm_attack import test_fgsm
from utils import plot_and_save_adversarial_examples, plot_accuracy_vs_epsilon

# Konstanten und Dateipfade
MODEL_PATH = "mnist_resnet50_model.h5"
SAVE_DIR = "results/"

# 1. MNIST-Datensatz laden
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Daten vorverarbeiten: Reshape, Normalisieren und Größe ändern
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Bilder auf 32x32 vergrößern für ResNet50
x_train = tf.image.resize(x_train, (32, 32))
x_test = tf.image.resize(x_test, (32, 32))

# Graustufenbilder in 3-Kanal-RGB-Bilder konvertieren
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)

# Labels in One-Hot-Encoding umwandeln
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Datenaugmentation für das Training
train_generator = ImageDataGenerator(
    rotation_range=40, 
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)
val_generator = ImageDataGenerator()

# Dateniteratoren erstellen
train_iterator = train_generator.flow(x_train, y_train, batch_size=512, shuffle=True)
val_iterator = val_generator.flow(x_test, y_test, batch_size=512, shuffle=False)

# 2. Modell erstellen oder laden
if not os.path.exists(MODEL_PATH):
    print("Modell nicht gefunden. Starte Training...")

    # ResNet50-Modell aufbauen
    model = Sequential([
        ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(32, 32, 3)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.layers[0].trainable = False  # ResNet50-Schichten einfrieren

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modell trainieren
    model.fit(train_iterator, validation_data=val_iterator, epochs=5)

    # Trainiertes Modell speichern
    model.save(MODEL_PATH)
    print(f"Modell gespeichert unter {MODEL_PATH}")
else:
    print("Trainiertes Modell gefunden. Lade gespeichertes Modell...")
    model = tf.keras.models.load_model(MODEL_PATH)

# 3. Modell evaluieren
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("\n--- Testergebnisse ---")
print(f"Testgenauigkeit: {test_accuracy * 100:.2f}%")
print(f"Testverlust: {test_loss:.4f}")

# 4. FGSM-Angriffe testen
epsilons = [0, 0.05, 0.1, 0.2, 0.3]
results = test_fgsm(model, x_test[:100], y_test[:100], epsilons)

# 5. Adversariale Beispiele und Genauigkeit vs. Epsilon plotten
plot_and_save_adversarial_examples(results, epsilons, SAVE_DIR)
plot_accuracy_vs_epsilon(results, epsilons, SAVE_DIR)
