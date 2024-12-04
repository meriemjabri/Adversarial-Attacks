# Analyse von Adversarialen Angriffen mit FGSM

## �bersicht
Dieses Projekt implementiert und analysiert adversariale Angriffe auf ein trainiertes Modell mithilfe der **Fast Gradient Sign Method (FGSM)**. Die Implementierung ist modular aufgebaut und kann leicht erweitert werden, um zus�tzliche Analysen durchzuf�hren.

## Projektstruktur

- **`model.py`**: Beinhaltet Funktionen, um das Modell zu trainieren oder ein bereits trainiertes Modell zu laden.
- **`fgsm_attack.py`**: Implementiert den FGSM-Angriff und bietet Werkzeuge zur Erstellung adversarialer Beispiele.
- **`utils.py`**: Stellt Hilfsfunktionen f�r die Datenvorverarbeitung, Evaluation und Visualisierung der Ergebnisse bereit.
- **`main.py`**: F�hrt die gesamte Analyse aus, inklusive Modelltraining, Angriffsausf�hrung und Ergebnisvisualisierung.
- **`results/`**: Verzeichnis f�r gespeicherte Ergebnisse und generierte Plots.
- **`fgsm_results.json`**: Enth�lt die gespeicherten Ergebnisse der FGSM-Angriffe.
- **`README.md`**: Dokumentation des Projekts mit Anweisungen zur Nutzung.

---

## Voraussetzungen

### Installieren der ben�tigten Pakete
F�hren Sie den folgenden Befehl aus, um die ben�tigten Python-Bibliotheken zu installieren:

```bash
pip install tensorflow matplotlib numpy
