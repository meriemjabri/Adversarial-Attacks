import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_and_save_adversarial_examples(results, epsilons, save_dir):
    """
    Plotten und Speichern von adversarialen Beispielen für gegebene Epsilon-Werte.
    Es wird sichergestellt, dass dieselben Eingabebilder über alle Epsilon-Werte verwendet werden.

    Parameter:
        results (dict): Dictionary mit Ergebnissen, das adversariale Beispiele enthält.
        epsilons (list): Liste der Epsilon-Werte.
        save_dir (str): Verzeichnis, in dem die geplotteten Bilder gespeichert werden sollen.

    Rückgabewert:
        None
    """
    num_epsilons = len(epsilons)
    num_images = min(len(results[epsilons[0]]["examples"]), 5)  # Maximal 5 Bilder für konsistente Vergleiche
    images_per_row = 5  # Anzahl der Bilder pro Zeile
    num_rows = (num_images // images_per_row) + (num_images % images_per_row > 0)

    plt.figure(figsize=(images_per_row * 4, num_epsilons * num_rows * 3))
    cnt = 0

    for i, epsilon in enumerate(epsilons):
        examples = results[epsilon]["examples"]

        for j in range(num_images):
            # Adversariales Beispiel für den aktuellen Epsilon-Wert extrahieren
            true_label, adv_pred, adv_confidence, ex = examples[j]

            # Berechnung der Positionen für Zeile und Spalte
            row = (j // images_per_row) + (i * num_rows)
            col = j % images_per_row

            # Subplot erstellen
            cnt += 1
            plt.subplot(num_epsilons * num_rows, images_per_row, cnt)
            plt.xticks([], [])  # Entferne x-Achsen-Beschriftungen
            plt.yticks([], [])  # Entferne y-Achsen-Beschriftungen
            if col == 0:  # Epsilon-Beschriftung in der ersten Spalte hinzufügen
                plt.ylabel(f"Eps: {epsilon:.2f}", fontsize=12)

            # Titel und Bild plotten
            plt.title(
                f"{true_label} -> {adv_pred}\nConf: {adv_confidence * 100:.2f}%",
                fontsize=10,
            )
            plt.imshow(ex.squeeze(), cmap="gray")

    plt.tight_layout()

    # Plot speichern
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "adversarial_examples.png"))
    plt.show()


def save_results_to_json(results, filename="fgsm_results.json"):
    """
    Speichert FGSM-Angriffs-Ergebnisse in einer JSON-Datei.

    Parameter:
        results (dict): Dictionary mit Genauigkeit und adversarialen Beispielen.
        filename (str): Name der Ausgabedatei im JSON-Format.

    Rückgabewert:
        None
    """
    # Ergebnisse in ein JSON-kompatibles Format umwandeln
    serializable_results = {
        float(epsilon): {
            "accuracy": float(result["accuracy"]),
            "examples": [
                {
                    "true_label": int(example[0]),
                    "adv_pred": int(example[1]),
                    "true_label_confidence": float(example[2]),
                    "perturbed_image": example[3].tolist()
                } for example in result["examples"]
            ]
        } for epsilon, result in results.items()
    }

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Ergebnisse wurden in {filename} gespeichert.")


def plot_accuracy_vs_epsilon(results, epsilons, save_dir=None):
    """
    Plotten der Genauigkeit vs. Epsilon-Werte und optionales Speichern des Diagramms.

    Parameter:
        results (dict): Dictionary mit FGSM-Ergebnissen und Genauigkeit für jeden Epsilon-Wert.
        epsilons (list): Liste der Epsilon-Werte.
        save_dir (str, optional): Verzeichnis, in dem das Diagramm gespeichert wird. Wenn None, wird nicht gespeichert.

    Rückgabewert:
        None
    """
    accuracies = [results[eps]["accuracy"] for eps in epsilons]

    plt.figure(figsize=(6, 5))
    plt.plot(epsilons, accuracies, "*-", label="Testgenauigkeit")

    # Achsenbeschriftungen und Titel hinzufügen
    plt.yticks(np.arange(0, 1.1, step=0.1))  # Y-Achse: Werte zwischen 0 und 1 in 0.1-Schritten
    plt.xticks(np.arange(0, max(epsilons) + 0.05, step=0.05))  # X-Achse: Werte zwischen 0 und max(epsilons)
    plt.title("Genauigkeit vs. Epsilon")
    plt.xlabel("Epsilon (Stärke der Störung)")
    plt.ylabel("Genauigkeit")

    # Raster und Legende hinzufügen
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Diagramm speichern, falls Verzeichnis angegeben
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "accuracy_vs_epsilon.png"))

    plt.show()
