# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import tensorflow as tf
from model import load_or_train_model
from fgsm_attack import test_fgsm
from utils import plot_and_save_adversarial_examples, plot_accuracy_vs_epsilon

# Konstanten und Dateipfade
MODEL_PATH = "mnist_resnet50_model.h5"
SAVE_DIR = "results/"
RESULTS_PATH = os.path.join(SAVE_DIR, "fgsm_results.json")

# 1. MNIST-Datensatz laden
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Daten vorverarbeiten: Reshape, Normalisieren und Größe ändern
x_train = tf.image.resize(x_train[..., None] / 255.0, (32, 32))
x_test = tf.image.resize(x_test[..., None] / 255.0, (32, 32))
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Modell laden oder trainieren
model = load_or_train_model(x_train, y_train, x_test, y_test)

# FGSM-Ergebnisse laden oder testen
if os.path.exists(RESULTS_PATH):
    print("FGSM-Ergebnisse gefunden. Lade gespeicherte Ergebnisse...")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
else:
    print("FGSM-Ergebnisse nicht gefunden. Starte Angriff...")
    epsilons = [0, 0.05, 0.1, 0.2, 0.3]
    results = test_fgsm(model, x_test[:100], y_test[:100], epsilons, RESULTS_PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)

# Adversariale Beispiele und Genauigkeit vs. Epsilon plotten
plot_and_save_adversarial_examples(results, epsilons, SAVE_DIR)
plot_accuracy_vs_epsilon(results, epsilons, SAVE_DIR)
