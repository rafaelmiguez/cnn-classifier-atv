import os
from PIL import Image
import numpy as np
from tensorflow import keras

from params import img_size, color_mode, test_model


def run():
    model = keras.models.load_model(f'tmp/{test_model}')
    labels = os.listdir('imgs')
    filenames = []
    imgs_arrs = []
    for filename in os.listdir('predict_imgs'):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = f'predict_imgs/{filename}'
            filenames.append(filename)
            with Image.open(img_path) as img:
                img = img.convert(color_mode)
                img = img.resize((img_size, img_size))
                img_arr = np.array(img)
                imgs_arrs.append(img_arr)

    predictions = model.predict(np.array(imgs_arrs))
    for filename, prediction in zip(filenames, predictions):
        best_prediction_idx = np.argmax(prediction)
        best_prediction = prediction[best_prediction_idx]
        print(
            f'{filename}: {labels[best_prediction_idx]} ({best_prediction * 100:.2f}%)')
