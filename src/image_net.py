import itertools

import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import InceptionV3, inception_v3

from src.config import base_configuration
from src.file_loader import File
import numpy as np

from util.threading import LockedIterator


class ImageNet:
    def __init__(self, file_loader: File):
        self.file_loader = file_loader
        self.layers = []
        self.inception = None

    # @property
    # def layers(self) -> list:
    #     layers = []
    #
    #     layers.append(Dense(
    #         base_configuration['sizes']['rnn_input'],
    #         input_shape=[299 * 299 * 3]
    #     ))
    #
    #     return layers

    @property
    def inception_model(self) -> InceptionV3:
        # Initialize with imagenet weights
        self.inception = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

        # Fix weights
        for layer in self.inception.layers:
            layer.trainable = False

        self.layers.append(BatchNormalization(axis=-1))
        self.layers.append(Dense(base_configuration['sizes']['rnn_input']))
        # TODO: regularizer and initializer
        # kernel_regularizer=self.regularizer,
        # kernel_initializer=self.initializer

        return self.inception, self.layers

    @staticmethod
    def preprocess_image(path):
        loaded_image = load_img(path, target_size=(299, 299))
        image_array = img_to_array(loaded_image)
        image_array = inception_v3.preprocess_input(image_array)
        return image_array

    @property
    def images(self):
        def generator():
            while True:
                print('RESTARTING IMAGE GENERATOR')
                for file_id, path in self.file_loader.id_file_map.items():
                    yield (file_id, self.preprocess_image(path))

        return LockedIterator(generator())

    @property
    def images_num(self):
        return len(self.file_loader.id_file_map)

    @property
    def captions_num(self):
        return sum((len(self.file_loader.id_caption_map[key]) for key in self.file_loader.id_file_map.keys()))


if __name__ == "__main__":
    # file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(None)

    model = Sequential()
    [model.add(layer) for layer in image_net.layers]


    model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))

    print(model.predict(np.random.normal(size=[1, 299 * 299 * 3])))
    #print(model.predict(np.asarray([image_net.preprocess_image("")])))

