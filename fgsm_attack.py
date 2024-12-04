# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import json

def fgsm_attack(image, label, model, epsilon):
    """
    Führt einen FGSM-Angriff auf ein einzelnes Bild durch.
    """
    image = tf.convert_to_tensor(image)

    # Berechnung des Gradienten
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction)
    gradient = tape.gradient(loss, image)

    # Generierung des adversarialen Beispiels
    perturbation = epsilon * tf.sign(gradient)
    perturbed_image = image + perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)  # Werte auf [0, 1] beschränken

    return perturbed_image


def test_fgsm(model, x_test, y_test, epsilons, results_path="fgsm_results.json"):
    """
    Testet FGSM-Angriffe für mehrere Epsilon-Werte und speichert Ergebnisse in einer JSON-Datei.
    """
    results = {}

    for epsilon in epsilons:
        print(f"Teste FGSM mit epsilon = {epsilon}")
        correct = 0
        adv_examples = []

        # Schleife über die Testdaten
        for i in range(len(x_test)):
            image = tf.convert_to_tensor(x_test[i:i + 1], dtype=tf.float32)
            label = tf.convert_to_tensor(y_test[i:i + 1], dtype=tf.float32)

            prediction = model(image)
            init_pred = tf.argmax(prediction, axis=1).numpy()[0]
            true_label = tf.argmax(label, axis=1).numpy()[0]

            if init_pred != true_label:  # Überspringe falsche ursprüngliche Vorhersagen
                continue

            perturbed_image = fgsm_attack(image, label, model, epsilon)
            adv_prediction = model(perturbed_image)
            adv_pred = tf.argmax(adv_prediction, axis=1).numpy()[0]

            if adv_pred == true_label:
                correct += 1

            adv_examples.append((
                true_label,
                adv_pred,
                adv_prediction[0][true_label].numpy(),
                perturbed_image.numpy()
            ))

        total = len(x_test)
        final_acc = correct / total
        print(f"Epsilon: {epsilon}\tGenauigkeit: {correct} / {total} = {final_acc:.4f}")

        results[epsilon] = {
            "accuracy": final_acc,
            "examples": adv_examples
        }

    # Ergebnisse in einer JSON-Datei speichern
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Ergebnisse gespeichert unter {results_path}")

    return results
