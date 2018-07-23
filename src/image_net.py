import itertools

import keras
from keras import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, BatchNormalization, RepeatVector
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

    @property
    def layers(self) -> list:
        layers = []

        #layers.append(BatchNormalization(axis=-1))
        layers.append(Dense(base_configuration['sizes']['rnn_input'],
                                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.1)))
        layers.append(RepeatVector(1))

        return layers

    @property
    def inception_model(self) -> InceptionV3:
        # Initialize with imagenet weights
        inception = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

        # Fix weights
        for layer in inception.layers:
            layer.trainable = False

        # TODO: regularizer and initializer
        # kernel_regularizer=self.regularizer,
        # kernel_initializer=self.initializer

        image_model, image_net_layers = inception, self.layers

        prev_output = image_model.output
        for layer in image_net_layers:
            prev_output = layer(prev_output)

        return image_model, prev_output

    @staticmethod
    def preprocess_image(path):
        loaded_image = load_img(path, target_size=(299, 299))
        tmp = img_to_array(loaded_image)
        return inception_v3.preprocess_input(tmp)

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

