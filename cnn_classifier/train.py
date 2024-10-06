import os
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from params import random_state, img_size, epochs, color_mode


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
    model = keras.models.Sequential([
        # Camadas de convolução
        keras.layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(img_size, img_size, 1 if color_mode == 'L' else 3)),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu'),

        # Linearizar matriz para vetor 1d
        keras.layers.Flatten(),

        # Camadas totalmente conectadas (dense)
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(os.listdir('imgs')))
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    print('Training model...')
    model.fit(train_imgs_arr, train_labels, epochs=epochs)

    _, test_acc = model.evaluate(
        test_imgs_arr,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    print('Exporting model...')
    model.save('tmp/model.keras')
