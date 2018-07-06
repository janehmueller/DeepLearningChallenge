import itertools
import json
from zipfile import ZipFile

import shutil

import os
import requests

from os import path
from typing import List, Dict
from itertools import groupby

import numpy as np
from keras import Sequential
from keras.engine import Layer
from keras.layers import Dense
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
        self._input_file = self._input_file if path.isabs(self._input_file) else path.join('.', self._input_file)
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


def word_embedding(vocab: Dict[str, int], word_vectors: WordVector):
    output_size = word_vectors.embedding_size()
    input_size = len(vocab) + 1

    sorted_vocab = list(vocab.items())
    sorted_vocab.sort(key=lambda x: x[1])

    word_vector_weights = []
    word_vector_weights.append(np.zeros(output_size))
    for caption, idx in sorted_vocab:
        caption_word_vector = word_vectors.vectorize_word(caption)

        if caption_word_vector is None:
            caption_word_vector = np.random.normal(size=output_size, scale=np.sqrt(2./(output_size+input_size)))

        word_vector_weights.append(caption_word_vector)

    biases = np.zeros(output_size)

    layer = Dense(output_size, input_shape=[input_size], weights=[np.asarray(word_vector_weights), biases])
    layer.trainable = False
    return [layer]


class TextPreprocessor(object):
    """
    Preprocessor that tokenizes and one-hot encodes captions.
    """
    vocab_file: str = "vocab.json"
    tokenizer: Tokenizer = None
    _vocab: Dict[str, int] = None
    inverse_vocab: Dict[int, str] = None

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.eos_token()])

    @staticmethod
    def eos_token() -> str:
        """
        The end-of-string token, which signifies the end of a caption.
        :return: the end-of-string token as string
        """
        return base_configuration["eos_token"]

    def vocab_size(self) -> int:
        """
        The number of words that the vocabulary contains.
        :return: the size as int
        """
        return len(self.vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @vocab.setter
    def vocab(self, value: Dict[str, int]):
        """
        Overwrites the vocabulary with the new vocabulary and also sets the inverse vocabulary.
        :param value: the new vocabulary
        """
        self._vocab = value
        self.inverse_vocab = dict([(v, k) for k, v in self._vocab.items()])

    def eos_token_index(self) -> int:
        """
        Returns the index of the end-of-string token in the vocabulary
        :return: the index as int
        """
        return self.vocab[self.eos_token()]

    def unzip_and_flatten_id_to_captions(self, id_to_captions: Dict[int, List[str]]):
        """
        Flattens the dictionary of the image ids and the list of captions into two equally long lists of the ids and
        the captions. The list of ids contains every id exactly num_captions_for_that_image times.
        :param id_to_captions: dict of image id to list of captions for that image
        :return: tuple of flattened list of image ids and list of captions
        """
        flat_id_to_captions = [(image_id, caption) for image_id, image_captions in id_to_captions.items() for caption in image_captions]
        return zip(*list(flat_id_to_captions))

    def zip_flat_id_to_captions(self, ids: List[int], encoded_captions: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """
        Groups the flattened list of image ids and encoded captions into a dictionary pointing from image id to a list
        of the one-hot encoded captions.
        :param ids: list of ids returned by self.unzip_and_flatten_id_to_captions
        :param encoded_captions: numpy array of the one-hot encoded captions
        :return:
        """
        flat_id_to_captions = zip(ids, encoded_captions)
        grouped_data = groupby(flat_id_to_captions, lambda kv_pair: kv_pair[0])
        return dict([(image_id, list(map(lambda x: x[1], captions))) for image_id, captions in grouped_data])

    def process_captions(self, captions: List[str]):
        flat_captions = list(itertools.chain(*captions))

        self.tokenizer.fit_on_texts(flat_captions)
        self.vocab = self.tokenizer.word_index

    def encode_caption(self, caption):
        return self.encode_captions([caption])

    def encode_captions(self, captions: List[str]) -> np.ndarray:
        captions_indices = self.tokenizer.texts_to_sequences(captions)
        captions_indices = [caption + [self.eos_token_index()] for caption in captions_indices]
        captions_indices = np.array(list(itertools.zip_longest(*captions_indices, fillvalue=0))).T

        max_idx = len(self.vocab) + 1
        return [self.one_hot_encode_caption(caption, max_idx) for caption in captions_indices]

    def decode_captions(self, one_hot_captions: np.ndarray) -> List[str]:
        """
        Decodes one-hot encoded captions into a list of captions as string.
        :param one_hot_captions: numpy array of the one-hot encoded captions in the shape
        (num_captions, size_of_longest_caption, vocab_size + 1)
        :return: list of decoded captions as string
        """
        decoded_captions = []

        indices = np.argmax(one_hot_captions, axis=2)
        indices[indices == self.eos_token_index()] = 0
        for caption in indices:
            decoded_captions.append([self.inverse_vocab[idx] for idx in caption if idx != 0])

        return [" ".join(caption) for caption in decoded_captions]

    def serialize(self):
        path = base_configuration["tmp_path"] + "/" + self.vocab_file
        with open(path, "w") as file:
            json.dump(self.vocab, file)

    def deserialize(self):
        path = base_configuration["tmp_path"] + "/" + self.vocab_file
        with open(path, "r") as file:
            self.vocab = json.load(file)
            self.tokenizer.word_index = self.vocab

    def word_embedding_layer(self, word_vector_type: str = "fasttext") -> List[Layer]:
        """
        Builds the embedding layer that is prepended to every RNN timestep.
        :param word_vector_type: which word embeddings should be used (default fasttext)
        :return: one-element list of the dense layer
        """

        word_vectors = WordVector(self.vocab, word_vector_type)
        output_size = word_vectors.embedding_size()
        input_size = self.vocab_size() + 1  # add 1 since the vocab does not contain 0 as index (so a vocab with size 1
                                            # needs a vector with 2 elements for its one-hot encoding)

        sorted_vocab = list(self.vocab.items())
        sorted_vocab.sort(key=lambda x: x[1])

        word_vector_weights = []
        word_vector_weights.append(np.zeros(output_size))
        for caption, idx in sorted_vocab:
            caption_word_vector = word_vectors.vectorize_word(caption)

            if caption_word_vector is None:
                caption_word_vector = np.random.normal(size=output_size, scale=np.sqrt(2. / (output_size + input_size)))

            word_vector_weights.append(caption_word_vector)

        biases = np.zeros(output_size)

        layer = Dense(output_size, input_shape=[input_size], weights=[np.asarray(word_vector_weights), biases])
        layer.trainable = False
        return [layer]


if __name__ == "__main__":
    with open("data/annotations/pretty_train.json") as file:
        data = json.load(file)
        data = [annotation["caption"] for annotation in data["annotations"]]

        tp = TextPreprocessor()
        tp.fit_captions(data)

        model = Sequential()
        [model.add(layer) for layer in tp.word_embedding_layer("fasttext")]

        model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))
        tmp = tp.encode_captions([TextPreprocessor.eos_token()])[0][0]
        print(model.predict(np.array([tmp])))
