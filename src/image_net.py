import keras
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array

from src.config import base_configuration
from src.file_loader import File
import numpy as np


class ImageNet:
    def __init__(self, file_loader: File):
        self.file_loader = file_loader

    @property
    def layers(self) -> list:
        layers = []

        layers.append(Dense(
            base_configuration['sizes']['rnn_input'],
            input_shape=[299 * 299 * 3]
        ))

        return layers

    @staticmethod
    def preprocess_image(path):
        loaded_image = load_img(path, (299, 299))
        return np.reshape(img_to_array(loaded_image), [299 * 299 * 3])

    @property
    def images(self):
        return ((file_id, self.preprocess_image(path)) for file_id, path in self.file_loader.id_file_map)


if __name__ == "__main__":
    #file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(None)

    model = Sequential()
    [model.add(layer) for layer in image_net.layers]


    model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))

    print(model.predict(np.random.normal(size=[1, 299 * 299 * 3])))
    #print(model.predict(np.asarray([image_net.preprocess_image("")])))

