import keras
from keras import Sequential
from keras.layers import Dense

from src.config import base_configuration
from src.file_loader import File


class ImageNet:
    def __init__(self, file_loader: File):
        self.file_loader = file_loader

    @property
    def layers(self) -> list:
        layers = []

        layers.append(Dense(
            base_configuration['sizes']['rnn_input'],
            input_shape=(299, 299, 3)
        ))

        return layers

    @staticmethod
    def preprocess_image(path):
        loaded_image = keras.preprocessing.image.load_img(path, (299, 299))
        return keras.preprocessing.image.img_to_array(loaded_image)

    @property
    def images(self):
        return (self.preprocess_image(path) for path in self.file_loader.id_file_map.values())
