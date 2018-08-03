import itertools
import math
from typing import Generator

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from os import path, makedirs
import time

from keras import Sequential, Model, Input
from keras.layers import Dense, TimeDistributed, Concatenate, BatchNormalization
import numpy as np
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.rnn_net import RNNNet
from src.text_preprocessing import TextPreprocessor
from util.checkGPU import onGPU, countGPU


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def training_data(images, text_preprocessor: TextPreprocessor, file_loader: File):
    batch_size = base_configuration['batch_size']
    image_shape = [299, 299, 3]
    batch_images = np.zeros(shape=[batch_size] + image_shape)

    caption_length = base_configuration['sizes']['repeat_vector_length']
    caption_output_length = caption_length + 1
    one_hot_size = text_preprocessor.one_hot_encoding_size
    batch_captions = np.zeros(shape=[batch_size, caption_output_length, one_hot_size])
    batch_input_captions = np.zeros(shape=[batch_size, caption_length])

    i = 0
    for image_id, image in images:
        for caption in file_loader.id_caption_map[image_id]:
            if i >= batch_size:
                # yield (np.copy(batch_images), np.copy(batch_captions)) PROBABLY WE SHOULD USE THIS
                yield ([batch_images, batch_input_captions], batch_captions)
                i = 0
            batch_images[i] = image
            batch_captions[i] = text_preprocessor.encode_caption(caption)
            batch_input_captions[i] = text_preprocessor.encode_caption(caption, one_hot=False)
            i += 1


def main():
    timestamp = str(round(time.time()))
    model_dir = path.join(base_configuration['tmp_path'], 'model-saves') #.' + timestamp)
    makedirs(model_dir, exist_ok=True)

    file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)
    rnn_net = RNNNet()
    text_preprocessor = TextPreprocessor()
    text_preprocessor.process_captions(file_loader.id_caption_map[key] for key in file_loader.id_file_map)
    text_preprocessor.serialize(model_dir)

    # Build Model Layers
    # Image Model
    image_model, image_embedding = image_net.inception_model
    image_input = image_model.input

    # Sentence Model
    sentence_input, sentence_embedding = text_preprocessor.word_embedding_layer()

    sequence_input = Concatenate(axis=1)([image_embedding, sentence_embedding])

    # RNN Here
    input_ = sequence_input
    for rnn_layer in rnn_net.layers:
        rnn_out = rnn_layer(input_)
        input_ = rnn_out

    sequence_output = TimeDistributed(Dense(text_preprocessor.one_hot_encoding_size, activation='softmax'))(rnn_out)

    model = Model(inputs=[image_input, sentence_input], outputs=sequence_output)

    if onGPU and countGPU is None:
        model = multi_gpu_model(model)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(clipnorm=5.0),
        **base_configuration['model_hyper_params']
    )

    training_data_generator = training_data(image_net.images, text_preprocessor, file_loader)

    checkpoint = ModelCheckpoint(path.join(model_dir, '{epoch:02d}.hdf5'), verbose=1)
    ###
    # in order to start tensorboard call:
    # tensorboard --logdir=logs/ --port=<any free port>
    ###
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    callbacks = [
        checkpoint,
        tensorboard
    ]

    step_size = math.ceil((image_net.captions_num / base_configuration['batch_size']))
    model.fit_generator(training_data_generator,
                        steps_per_epoch=step_size,
                        callbacks=callbacks,
                        use_multiprocessing=False,
                        workers=1,
                        **base_configuration['fit_params'])

    model.save(path.join(model_dir, 'model-all.hdf5'), overwrite=True)


if __name__ == '__main__':
    main()
