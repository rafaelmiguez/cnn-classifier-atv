# CNN Image Classifier

A Python project for training and evaluating convolutional neural networks (CNNs) for image classification. Developed as part of an Advanced Topics in Scientific Initiation course.

## Overview

The project provides a reproducible workflow for preparing image datasets, training a CNN from scratch, fine-tuning a model with transfer learning, and running predictions on new images.

## Technology

- Python
- TensorFlow / Keras
- Poetry
- NumPy and image-processing utilities

## Dataset layout

Training images should be organized in class-specific subdirectories:

```
imgs/
├── class_a/
├── class_b/
└── class_c/
```

Place images to classify in `predict_imgs/`. The folder names inside `imgs/` define the target classes.

## Configuration

Use `params.py` to configure preprocessing and training, including:

- Image color mode and target size
- Optimizer and loss function
- Number of training and fine-tuning epochs
- Number of layers to unfreeze during fine-tuning
- Model selected for prediction

## Running the project

Install the dependencies with Poetry:

```bash
poetry install
```

Preprocess the training images:

```bash
poetry run preprocess
```

Train a CNN from scratch or fine-tune a model:

```bash
poetry run train
poetry run fine_tune
```

Run predictions on the images in `predict_imgs/`:

```bash
poetry run predict
```

Generated models and intermediate files are written to the project's `tmp/` directory according to the selected configuration.

## Project structure

- `cnn_classifier/` — preprocessing, training, fine-tuning, and prediction modules
- `imgs/` — training dataset organized by class
- `predict_imgs/` — images used for inference
- `params.py` — experiment configuration
- `ROTEIRO_TENSORFLOW.md` and `ROTEIRO_FINE_TUNNING.md` — supporting experiment notes

## Notes

This repository contains an academic machine-learning project and is intended for study and experimentation. Model performance depends on the dataset, preprocessing choices, and training configuration.
