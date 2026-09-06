# CNN Image Classifier

An academic coursework fork for exploring image classification with Python, TensorFlow/Keras, and Poetry.

## Origin and attribution

This repository is based on [RafaSantos484/cnn-classifier](https://github.com/RafaSantos484/cnn-classifier). The original project provides the foundation for the training, fine-tuning, and prediction workflows documented below. This fork is retained as a record of coursework and learning; it does not claim independent authorship of the original implementation. See the [license](./LICENSE) and repository history for attribution.


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
