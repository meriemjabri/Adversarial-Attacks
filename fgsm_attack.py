# -*- coding: utf-8 -*-
import tensorflow as tf
from utils import save_results_to_json  # Importiere die Funktion aus utils.py


def fgsm_attack(image, label, model, epsilon):
    """
    Führt einen FGSM-Angriff auf ein einzelnes Bild durch.

    Parameter:
        image (tf.Tensor): Eingabebild.
        label (tf.Tensor): Wahre Klasse des Bildes (One-Hot-Encoding).
        model (tf.keras.Model): Modell, das angegriffen werden soll.
        epsilon (float): Stärke der Störung.

    Rückgabewert:
        tf.Tensor: Adversariales Beispiel mit Störung.
    """
    image = tf.convert_to_tensor(image)

    # 1. Berechnung des Gradienten
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction)
    gradient = tape.gradient(loss, image)

    # 2. Generierung des adversarialen Beispiels
    perturbation = epsilon * tf.sign(gradient)

    # 3. Aktualisierung der Eingabe
    perturbed_image = image + perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)  # Werte auf den Bereich [0, 1] beschränken

    return perturbed_image


def test_fgsm(model, x_test, y_test, epsilons):
    """
    Testet FGSM-Angriffe für mehrere Epsilon-Werte und bewertet die Robustheit des Modells.

    Parameter:
        model (tf.keras.Model): Modell, das getestet wird.
        x_test (np.ndarray): Testdaten (Bilder).
        y_test (np.ndarray): Testlabels (One-Hot-Encoding).
        epsilons (list): Liste der Epsilon-Werte für FGSM.

    Rückgabewert:
        dict: Ergebnisse mit Genauigkeit und adversarialen Beispielen für jeden Epsilon-Wert.
    """
    results = {}

    for epsilon in epsilons:
        print(f"Teste FGSM mit epsilon = {epsilon}")
        correct = 0
        adv_examples = []

        # Schleife über die Testdaten
        for i in range(len(x_test)):
            # Extrahiere Bild und Label
            image = tf.convert_to_tensor(x_test[i:i + 1], dtype=tf.float32)
            label = tf.convert_to_tensor(y_test[i:i + 1], dtype=tf.float32)

            # Vorhersage des Modells
            prediction = model(image)
            init_pred = tf.argmax(prediction, axis=1).numpy()[0]
            true_label = tf.argmax(label, axis=1).numpy()[0]

            # Überspringe, wenn die ursprüngliche Vorhersage falsch ist
            if init_pred != true_label:
                continue

            # Generiere ein adversariales Beispiel
            perturbed_image = fgsm_attack(image, label, model, epsilon)

            # Adversariale Vorhersage
            adv_prediction = model(perturbed_image)
            adv_pred = tf.argmax(adv_prediction, axis=1).numpy()[0]

            # Aktualisiere korrekte Vorhersagen
            if adv_pred == true_label:
                correct += 1

            # Speichere adversariales Beispiel
            adv_examples.append((
                true_label,
                adv_pred,
                adv_prediction[0][true_label].numpy(),
                perturbed_image.numpy()
            ))

        # Berechne die Genauigkeit für den aktuellen Epsilon-Wert
        total = len(x_test)
        final_acc = correct / total
        print(f"Epsilon: {epsilon}\tGenauigkeit: {correct} / {total} = {final_acc:.4f}")

        # Ergebnisse speichern
        results[epsilon] = {
            "accuracy": final_acc,
            "examples": adv_examples
        }

    # Ergebnisse in einer JSON-Datei speichern
    save_results_to_json(results)  # Aufruf der Funktion aus utils.py

    return results
