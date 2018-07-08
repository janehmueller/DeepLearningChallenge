import itertools
from typing import Generator

from keras.callbacks import ModelCheckpoint
from os import path, makedirs
import time

from keras import Sequential, Model
from keras.layers import Dense, TimeDistributed
import numpy as np
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits

from src.config import base_configuration
from src.image_net import ImageNet
from src.rnn_net import RNNNet
from util.loss import categorical_crossentropy_from_logits
from util.checkGPU import onGPU
from util.batch_sequence import BatchSequence


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def main():
    timestamp = str(round(time.time()))
    model_dir = path.join(base_configuration['tmp_path'], 'model-saves.' + timestamp)
    makedirs(model_dir, exist_ok=True)

    batch_sequence = BatchSequence(model_dir)

    image_net = ImageNet()
    rnn_net = RNNNet()
    model = Sequential()
    inception, image_net_layers = image_net.inception_model
    model_list_add(model, image_net_layers)
    # model_list_add(model, text_preprocessor.word_embedding_layer()))
    model_list_add(model, rnn_net.layers)
    model.add(TimeDistributed(Dense(batch_sequence.text_preprocessor.one_hot_encoding_size, activation='relu')))

    # model = multi_gpu_model(model)

    model.compile(loss=categorical_crossentropy_from_logits, **base_configuration['model_hyper_params'])

    func_model = model(inception.output)
    model = Model(inputs=inception.input, outputs=func_model)

    if onGPU:
        model = multi_gpu_model(model)
    model.compile(loss=categorical_crossentropy_from_logits, **base_configuration['model_hyper_params'])

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

    model.fit_generator(batch_sequence,
                        callbacks=callbacks,
                        use_multiprocessing=False,
                        workers=1,
                        **base_configuration['fit_params'])

    model.save(path.join(model_dir, 'model-all.hdf5'), overwrite=True)


if __name__ == '__main__':
    main()
