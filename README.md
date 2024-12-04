# Adversarial Attack Analysis with FGSM

## Overview
This project implements and analyzes adversarial attacks on a trained model using the Fast Gradient Sign Method (FGSM) attack. The implementation is designed to be modular and easy to extend for further analysis.

## Files
1. **model.py**: Contains functions to train or load the model used for adversarial attack analysis.
2. **fgsm_attack.py**: Implements the FGSM attack and provides utilities for crafting adversarial examples.
3. **utils.py**: Provides helper functions for data preprocessing, evaluation, and visualization.
4. **README.md**: Documentation for the project, including usage instructions and an overview of the structure.

## Usage

### Install Required Packages
Run the following command to install dependencies:
```bash
pip install tensorflow matplotlib numpy
