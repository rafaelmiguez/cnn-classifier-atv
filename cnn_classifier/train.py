import os
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import layers, models

from params import random_state, epochs


le = LabelEncoder()


def load_imgs_arr_labels(folder: str):
    imgs_arr = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        imgs_arr.append(np.array(img))

    label = folder.split('/')[-1]
    return np.array(imgs_arr), np.array([label] * len(imgs_arr))


def run():
    print('Reading images...')
    imgs_arr: NDArray = None
    imgs_labels: NDArray = None
    for path in os.listdir('tmp/processed_imgs'):
        arr, labels = load_imgs_arr_labels(f'tmp/processed_imgs/{path}')
        if imgs_arr is None:
            imgs_arr, imgs_labels = arr, labels
        else:
            imgs_arr = np.concatenate([imgs_arr, arr])
            imgs_labels = np.concatenate([imgs_labels, labels])

    print('Encoding labels...')
    encoded_imgs_labels = le.fit_transform(imgs_labels)

    train_imgs_arr, test_imgs_arr, train_labels, test_labels = train_test_split(
        imgs_arr, encoded_imgs_labels, test_size=0.2, random_state=random_state)
    train_imgs_arr = train_imgs_arr / 255.0
    test_imgs_arr = test_imgs_arr / 255.0

    print('Creating model...')
    num_classes = len(os.listdir('imgs'))
    model = models.Sequential()

    # Camadas de convolução
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
              input_shape=train_imgs_arr.shape[1:]))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Linearização de array de output
    model.add(layers.Flatten())
    # Camadas finais densas
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Training model...')
    model.fit(train_imgs_arr, train_labels, epochs=epochs)

    _, test_acc = model.evaluate(test_imgs_arr, test_labels)
    print('\nTest accuracy:', test_acc)

    print('Exporting model...')
    model.save('tmp/model.keras')
