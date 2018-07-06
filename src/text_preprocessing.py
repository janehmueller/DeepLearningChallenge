import itertools
import json
import os
import requests
import shutil
from os import path
from typing import List, Dict
from zipfile import ZipFile

import numpy as np
from keras import Sequential
from keras.layers import Embedding, Dense
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer

from src.config import base_configuration


class WordVector(object):
    def vectorize_word(self, word: str):
        return self.word_vectors[word]

    def embedding_size(self):
        return 300

    def __init__(self, vocab, vector_type):
        if vector_type not in base_configuration['files']['pretrained']['word_vectors']:
            raise ValueError('Invalid vector type {}'.format(vector_type))

        self._vector_type = vector_type
        self._input_file = self.vector_type_info['path']
        self._input_file = path.join(base_configuration['pretrained_models_path'], self._input_file)
        self._input_file = self._input_file if os.path.isabs(self._input_file) else os.path.join('.', self._input_file)
        self._fetch_data_if_needed()
        self._load_pretrained_vectors(vocab)

    @property
    def vector_type_info(self):
        return base_configuration['files']['pretrained']['word_vectors'][self._vector_type]

    def _load_pretrained_vectors(self, vocab):
        with open(self._input_file, "r", encoding="utf-8") as file:
            if self._vector_type == 'fasttext':
                next(file)  # Skip the first line as it is a header, this is format specific
            self._load_pretrained_vectors_file(file, vocab)

    def _fetch_data_if_needed(self):
        if not path.exists(self._input_file):
            os.makedirs(path.dirname(self._input_file), exist_ok=True)

            uri_info = self.vector_type_info['uri']

            if uri_info['type'] == 'file':
                print("Downloading {} to {}".format(uri_info['url'], self._input_file))
                response = requests.get(uri_info['url'], stream=True)
                with open(self._input_file, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
                del response
            elif uri_info['type'] == 'zip':
                print("Downloading {} to {}.zip".format(uri_info['url'], self._input_file))
                zip_file = self._input_file + '.zip'
                response = requests.get(uri_info['url'], stream=True)
                with open(zip_file, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
                del response

                with ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(path.dirname(self._input_file))
                os.unlink(zip_file)

    def _load_pretrained_vectors_file(self, file_obj, vocab: Dict[str, int]):
        self.word_vectors = {}
        unsupported_lines = []

        for line in file_obj:
            tokens = line.split()
            word = " ".join(tokens[:-300])
            vector = tokens[-300:]
            if word == '.':
                self.word_vectors[TextPreprocessor.eos_token()] = np.asarray(vector, dtype='float32')
            elif word in vocab:
                if len(tokens) <= 300:
                    unsupported_lines.append(line)
                    continue
                self.word_vectors[word] = np.asarray(vector, dtype='float32')
        assert len(self.word_vectors) == len(vocab)
        assert TextPreprocessor.eos_token() in self.word_vectors
        assert len(unsupported_lines) <= 0, "{} unsupported lines\n".format(len(unsupported_lines)) + "\n".join(unsupported_lines)


def word_embedding(vocab:Dict[str, int], word_vectors:WordVector):
    output_size = word_vectors.embedding_size()
    input_size = len(vocab) + 1

    sorted_vocab = list(vocab.items())
    sorted_vocab.sort(key=lambda x: x[1])

    word_vector_weights = []
    word_vector_weights.append(np.zeros(output_size))
    for caption , idx in sorted_vocab:
        caption_word_vector = word_vectors.vectorize_word(caption)

        if caption_word_vector is None:
            caption_word_vector = np.random.normal(size=output_size, scale=np.sqrt(2./(output_size+input_size)))

        word_vector_weights.append(caption_word_vector)

    biases = np.zeros(output_size)

    layer = Dense(output_size, input_shape=[input_size], weights=[np.asarray(word_vector_weights), biases])
    layer.trainable = False
    return [layer]


class TextPreprocessor(object):
    vocab_file: str = "vocab.json"
    tokenizer: Tokenizer = None
    _dictionary: Dict[str, int] = None
    inverse_dictionary: Dict[int, str] = None

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.eos_token()])

    @staticmethod
    def eos_token():
        return "zeornd"

    def vocab_size(self):
        return len(self.dictionary)

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value
        self.inverse_dictionary = dict([(v, k) for k, v in self._dictionary.items()])

    def eos_token_index(self):
        return self.dictionary[self.eos_token()]

    def process_captions(self, captions: List[str]):
        self.tokenizer.fit_on_texts(captions)
        self.dictionary = self.tokenizer.word_index

    def encode_captions(self, captions: List[str]):
        captions_indices = self.tokenizer.texts_to_sequences(captions)
        captions_indices = [caption + [self.eos_token_index()] for caption in captions_indices]
        captions_indices = np.array(list(itertools.zip_longest(*captions_indices, fillvalue=0))).T

        max_idx = len(self.dictionary) + 1

        one_hot_captions = []
        for i, caption in enumerate(captions_indices):
            one_hot_captions.append([])
            for token in caption:
                one_hot = np.zeros(max_idx)
                one_hot[token] = int(bool(token)) # padding will be array of 0s
                one_hot_captions[i].append(one_hot)

        return np.asarray(one_hot_captions)

    def decode_captions(self, one_hot_captions):
        decoded_captions = []

        indices = np.argmax(one_hot_captions, axis=2)
        indices[indices == self.eos_token_index()] = 0
        for caption in indices:
            decoded_captions.append([self.inverse_dictionary[idx] for idx in caption if idx != 0])

        return [" ".join(caption) for caption in decoded_captions]

    def serialize(self):
        path = base_configuration["tmp_path"] + "/" + self.vocab_file
        with open(path, "w") as file:
            json.dump(self.dictionary, file)

    def deserialize(self):
        path = base_configuration["tmp_path"] + "/" + self.vocab_file
        with open(path, "r") as file:
            self.dictionary = json.load(file)
            self.tokenizer.word_index = self.dictionary


if __name__ == "__main__":

    with open("data/annotations/pretty_train.json") as file:
        data = json.load(file)
        data = [annotation["caption"] for annotation in data["annotations"]][:2]

        tp = TextPreprocessor()
        tp.process_captions(data)

        wv = WordVector(tp.dictionary, "fasttext")

        model = Sequential()
        [model.add(layer) for layer in word_embedding(tp.dictionary, wv)]

        model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))
        tmp = tp.encode_captions([TextPreprocessor.eos_token()])[0][0]
        print(model.predict(np.array([tmp])))
