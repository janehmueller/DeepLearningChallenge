from os import path

import numpy as np
from keras.models import load_model

from src.config import base_configuration


def main():
    model_dir = path.join(base_configuration['tmp_path'], 'model-saves')
    model_epoch = 5
    model_path = path.join(model_dir, '{:02d}.hdf5'.format(model_epoch))

    model = load_model('model.h5')

    samples = [np.array(42)]
    model.predict(samples)


if __name__ == '__main__':
    main()
