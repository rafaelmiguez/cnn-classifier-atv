import os
from PIL import Image
import numpy as np
from tensorflow import keras

from params import img_size, color_mode


def run():
    model = keras.models.load_model('tmp/model.keras')
    # imgs_arr: NDArray = None
    labels = os.listdir('imgs')
    results = []
    for filename in os.listdir('predict_imgs'):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = f'predict_imgs/{filename}'
            with Image.open(img_path) as img:
                img = img.convert(color_mode)
                img = img.resize((img_size, img_size))
                prediction = model.predict(np.array([np.array(img)]))
                results.append(f'{filename}: {labels[np.argmax(prediction)]}')

    for result in results:
        print(result)
