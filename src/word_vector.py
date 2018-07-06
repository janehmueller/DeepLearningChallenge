import numpy as np
import shutil
from os import path, makedirs, unlink
from typing import Dict
from zipfile import ZipFile

import requests

from src.config import base_configuration
from src.text_preprocessing import TextPreprocessor


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
            makedirs(path.dirname(self._input_file), exist_ok=True)

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
                unlink(zip_file)

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
