import itertools

from keras.callbacks import ModelCheckpoint
from os import path, makedirs
import time

from keras import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import multi_gpu_model

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.rnn_net import RNNNet
from src.text_preprocessing import TextPreprocessor


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def training_data(images, text_preprocessor, file_loader):
    batch_size = base_configuration['batch_size']
    batch_images = []
    batch_captions = []
    for image_id, image in images:
        for caption in file_loader.id_caption_map[image_id]:
            if len(batch_images) >= batch_size:
                yield (np.asarray(batch_images), np.asarray(batch_captions))
                batch_images = []
                batch_captions = []
            batch_images.append(image)
            batch_captions.append(text_preprocessor.encode_caption(caption))


def main():
    timestamp = str(round(time.time()))
    model_dir = path.join(base_configuration['tmp_path'], 'model-saves.' + timestamp)
    makedirs(model_dir, exist_ok=True)

    file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)
    rnn_net = RNNNet()
    text_preprocessor = TextPreprocessor()
    text_preprocessor.process_captions(file_loader.id_caption_map.values())

    model = Sequential()
    model_list_add(model, image_net.layers)
    # model_list_add(model, text_preprocessor.word_embedding_layer()))
    model_list_add(model, rnn_net.layers)
    model.add(Dense(len(text_preprocessor.vocab) + 1))

    model = multi_gpu_model(model)

    model.compile(**base_configuration['model_hyper_params'])

    training_data_generator = training_data(image_net.images, text_preprocessor, file_loader)

    checkpoint = ModelCheckpoint(path.join(model_dir, '{epoch:02d}.hdf5'), verbose=1)
    callbacks = [
        checkpoint,
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
