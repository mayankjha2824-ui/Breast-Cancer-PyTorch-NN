# Breast Cancer Detection using a Custom PyTorch Neural Network

This repository contains my first implementation of a basic Neural Network (Logistic Regression) built from scratch using PyTorch. The model is trained to classify tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Project Overview

Instead of relying on high-level PyTorch modules like `torch.nn`, this project implements the core mechanics of a neural network manually to gain a deeper understanding of deep learning fundamentals. 

Key features include:
* Manual weight and bias initialization using PyTorch tensors.
* Custom forward pass with a Sigmoid activation function.
* Custom Binary Cross-Entropy (BCE) loss function.
* Manual backpropagation and gradient descent updates.

## Dataset

The model uses the [Breast Cancer Wisconsin Dataset](https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv). 
* Features are standardized using `StandardScaler`.
* Target labels (Malignant/Benign) are encoded using `LabelEncoder`.

## Requirements

To run this project, you will need Python installed along with the following libraries:
* `torch`
* `pandas`
* `numpy`
* `scikit-learn`

You can install the dependencies using:
```bash
pip install -r requirements.txt
