# Analyse von Adversarialen Angriffen mit FGSM

## Übersicht
Dieses Projekt implementiert und analysiert adversariale Angriffe auf ein trainiertes Modell mithilfe der **Fast Gradient Sign Method (FGSM)**. Die Implementierung ist modular aufgebaut und kann leicht erweitert werden, um zusätzliche Analysen durchzuführen.

## Projektstruktur

- **`model.py`**: Beinhaltet Funktionen, um das Modell zu trainieren oder ein bereits trainiertes Modell zu laden.
- **`fgsm_attack.py`**: Implementiert den FGSM-Angriff und bietet Werkzeuge zur Erstellung adversarialer Beispiele.
- **`utils.py`**: Stellt Hilfsfunktionen für die Datenvorverarbeitung, Evaluation und Visualisierung der Ergebnisse bereit.
- **`main.py`**: Führt die gesamte Analyse aus, inklusive Modelltraining, Angriffsausführung und Ergebnisvisualisierung.
- **`results/`**: Verzeichnis für gespeicherte Ergebnisse und generierte Plots.
- **`fgsm_results.json`**: Enthält die gespeicherten Ergebnisse der FGSM-Angriffe.
- **`README.md`**: Dokumentation des Projekts mit Anweisungen zur Nutzung.

---

## Voraussetzungen

### Installieren der benötigten Pakete
Führen Sie den folgenden Befehl aus, um die benötigten Python-Bibliotheken zu installieren:

```bash
pip install tensorflow matplotlib numpy
