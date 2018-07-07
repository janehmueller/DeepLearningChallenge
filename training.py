import itertools
from typing import Generator

from keras.callbacks import ModelCheckpoint
from os import path, makedirs
import time

from keras import Sequential, Model, Input
from keras.layers import Dense, TimeDistributed, Concatenate, RepeatVector
import numpy as np
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.rnn_net import RNNNet
from src.text_preprocessing import TextPreprocessor
from util.loss import categorical_crossentropy_from_logits


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def training_data(images, text_preprocessor: TextPreprocessor, file_loader: File):
    batch_size = base_configuration['batch_size']
    image_shape = [299, 299, 3]
    batch_images = np.zeros(shape=[batch_size] + image_shape)
    caption_length = base_configuration['sizes']['repeat_vector_length']
    one_hot_size = text_preprocessor.one_hot_encoding_size
    batch_output_captions = np.zeros(shape=[batch_size, caption_length, one_hot_size])
    i = 0
    for image_id, image in images:
        for caption in file_loader.id_caption_map[image_id]:
            if i >= batch_size:
                # yield (np.copy(batch_images), np.copy(batch_captions)) PROBABLY WE SHOULD USE THIS
                yield ([batch_images, batch_output_captions[0]], batch_output_captions)
                i = 0
            batch_images[i] = image
            batch_output_captions[i] = text_preprocessor.encode_caption(caption)
            i += 1


def main():
    timestamp = str(round(time.time()))
    model_dir = path.join(base_configuration['tmp_path'], 'model-saves.' + timestamp)
    makedirs(model_dir, exist_ok=True)

    file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)
    rnn_net = RNNNet()
    text_preprocessor = TextPreprocessor()
    text_preprocessor.process_captions(file_loader.id_caption_map.values())
    text_preprocessor.serialize(model_dir)

    # Image model that has the InceptionV3 as input and outputs an RNN input size sized vector
    image_model = Sequential()
    inception, image_net_layers = image_net.inception_model
    inception_input = inception.input
    model_list_add(image_model, image_net_layers)
    func_image_model = image_model(inception.output)

    # Word embedding model that has one-hot encoding as input and outputs an RNN input size sized vector
    # base_configuration['sizes']['repeat_vector_length']
    sentence_input = Input(shape=[text_preprocessor.one_hot_encoding_size])
    sentence_model = text_preprocessor.word_embedding_layer()(sentence_input)

    # Concatenation of image and word embedding models that is the input of the RNN model
    func_rnn_input = Concatenate(axis=1)([func_image_model, sentence_model])

    # RNN model that outputs time-step many predictions of captions
    # rnn_model = Sequential()
    func_rnn_model = RepeatVector(base_configuration['sizes']['repeat_vector_length'])(func_rnn_input)
    func_rnn_model = rnn_net.GRUclass(base_configuration['sizes']['rnn_output'],
                     return_sequences=True,
                     dropout=.2,
                     recurrent_dropout=.2)(func_rnn_model)
    # model_list_add(rnn_model, rnn_net.layers)
    # func_rnn_model = rnn_model(func_rnn_input)
    func_rnn_model = TimeDistributed(Dense(text_preprocessor.one_hot_encoding_size, activation='relu'))(func_rnn_model)
    # rnn_model.add(TimeDistributed(Dense(text_preprocessor.one_hot_encoding_size, activation='relu')))
    # rnn_model.add(rnn_input)

    model = Model(inputs=[inception_input, sentence_input], outputs=func_rnn_model)

    model = multi_gpu_model(model)
    model.compile(loss=categorical_crossentropy_from_logits, **base_configuration['model_hyper_params'])

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

    step_size = int((image_net.captions_num / base_configuration['batch_size']) + .5)
    model.fit_generator(training_data_generator,
                        steps_per_epoch=step_size,
                        callbacks=callbacks,
                        use_multiprocessing=False,
                        workers=0,
                        **base_configuration['fit_params'])

    model.save(path.join(model_dir, 'model-all.hdf5'), overwrite=True)


if __name__ == '__main__':
    main()
