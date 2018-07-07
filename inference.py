from os import path

import numpy as np
from keras.models import load_model

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet


def prediction_data(images, file_loader):
    batch_size = base_configuration['batch_size']
    batch_images = []
    for image_id, image in images:
        if len(batch_images) >= batch_size:
            yield np.asarray(batch_images)
            batch_images = []
        batch_images.append(image)

def main():
    model_dir = path.join(base_configuration['tmp_path'], 'model-saves')
    model_epoch = 5
    model_path = path.join(model_dir, '{:02d}.hdf5'.format(model_epoch))

    model = load_model(model_path)

    file_loader = File.load(base_configuration['selected_dataset'])
    image_net = ImageNet(file_loader)

    prediction_data_generator = prediction_data(image_net.images, file_loader)

    model.predict_generator(prediction_data_generator)


if __name__ == '__main__':
    main()
