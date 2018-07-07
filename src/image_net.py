import itertools

import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
from keras.applications import InceptionV3

from src.config import base_configuration
import numpy as np



class ImageNet:
    def __init__(self):
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

if __name__ == "__main__":
    image_net = ImageNet()

    model = Sequential()
    [model.add(layer) for layer in image_net.layers]


    model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))

    print(model.predict(np.random.normal(size=[1, 299 * 299 * 3])))
    #print(model.predict(np.asarray([image_net.preprocess_image("")])))

