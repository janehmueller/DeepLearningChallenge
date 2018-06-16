import os
import shutil
from zipfile import ZipFile

import requests
from os import path

import numpy as np
from keras import backend as K, initializers

from util.preprocessors import CaptionPreprocessor
from util.config import base_configuration


class WordVector(object):
    """An initializer for Embedding layers of Keras.

    Inspired by https://github.com/danieljl/keras-image-captioning/blob/master/keras_image_captioning/word_vectors.py
    """

    def __init__(self, vocab_words, initializer, vector_type):
        self._vocab_words = set(vocab_words)
        self._word_vector_of = dict()

        if vector_type not in base_configuration['files']['pretrained']['word_vectors']:
            raise ValueError('Invalid vector type {}'.format(vector_type))

        self._vector_type = vector_type
        self._embedding_size = self.vector_type_info['embedding_size']
        self._input_file = self.vector_type_info['path']
        self._input_file = path.join(base_configuration['pretrained_models_path'], self._input_file)
        self._initializer = initializers.get(initializer)
        self._fetch_data_if_needed()
        self._load_pretrained_vectors()

    @property
    def vector_type_info(self):
        return base_configuration['files']['pretrained']['word_vectors'][self._vector_type]

    def _load_pretrained_vectors(self):
        with open(self._input_file) as file:
            if self._vector_type == 'fasttext':
                next(file)  # Skip the first line as it is a header, this is format specific
            self._load_pretrained_vectors_file(file)

    def _fetch_data_if_needed(self):
        if not path.exists(self._input_file):
            os.makedirs(path.dirname(self._input_file), exist_ok=True)

            uri_info = self.vector_type_info['uri']

            if uri_info['type'] == 'file':
                print("Downloading {}".format(uri_info['url']))
                response = requests.get(uri_info['url'], stream=True)
                with open(self._input_file, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
                del response
            elif uri_info['type'] == 'zip':
                zip_file = self._input_file + '.zip'
                response = requests.get(uri_info['url'], stream=True)
                with open(zip_file, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
                del response

                with ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(path.dirname(self._input_file))
                os.unlink(zip_file)

    def vectorize_words(self, words):
        vectors = []
        for word in words:
            vector = self._word_vector_of.get(word)
            vectors.append(vector)

        num_unknowns = len(filter(lambda x: x is None, vectors))
        inits = self._initializer(shape=(num_unknowns, self._embedding_size))
        inits = K.get_session().run(inits)
        inits = iter(inits)
        for i in range(len(vectors)):
            if vectors[i] is None:
                vectors[i] = next(inits)

        return np.array(vectors)

    def _load_pretrained_vectors_file(self, file_obj):
        for line in file_obj:
            tokens = line.split()
            word = tokens[0]
            if word == '.':
                self._word_vector_of[CaptionPreprocessor.EOS_TOKEN] = np.asarray(tokens[1:], dtype='float32')
            elif word in self._vocab_words:
                self._word_vector_of[word] = np.asarray(tokens[1:], dtype='float32')
        assert CaptionPreprocessor.EOS_TOKEN in self._word_vector_of
