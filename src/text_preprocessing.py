import itertools
import json
from typing import List, Dict
from itertools import groupby

import numpy as np
from keras import Sequential
from keras.engine import Layer
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer

from src.config import base_configuration
from src.word_vector import WordVector


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
        return "zeornd"

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
        flat_id_to_captions = [(image_id, caption) for image_id, image_captions in id_to_captions.items() for caption in image_captions]
        return zip(*list(flat_id_to_captions))

    def zip_flat_id_to_captions(self, ids: List[int], encoded_captions: np.ndarray) -> Dict[int, np.ndarray]:
        flat_id_to_captions = zip(ids, encoded_captions)
        grouped_data = groupby(flat_id_to_captions, lambda kv_pair: kv_pair[0])
        return dict([(image_id, np.asarray(list(map(lambda x: x[1], captions)))) for image_id, captions in grouped_data])

    def process_id_to_captions(self, id_to_captions: Dict[int, List[str]]) -> Dict[int, np.ndarray]:
        ids, captions = self.unzip_and_flatten_id_to_captions(id_to_captions)
        encoded_captions = self.process_captions(captions)
        return self.zip_flat_id_to_captions(ids, encoded_captions)

    def fit_on_id_to_captions(self, id_to_captions: Dict[int, List[str]]):
        ids, captions = self.unzip_and_flatten_id_to_captions(id_to_captions)
        self.fit_captions(captions)

    def encode_id_to_captions(self, id_to_captions: Dict[int, List[str]]) -> Dict[int, np.ndarray]:
        ids, captions = self.unzip_and_flatten_id_to_captions(id_to_captions)
        encoded_captions = self.encode_captions(captions)
        return self.zip_flat_id_to_captions(ids, encoded_captions)

    def process_captions(self, captions: List[str]) -> np.ndarray:
        self.fit_captions(captions)
        return self.encode_captions(captions)

    def fit_captions(self, captions: List[str]):
        self.tokenizer.fit_on_texts(captions)
        self.vocab = self.tokenizer.word_index

    def encode_captions(self, captions: List[str]) -> np.ndarray:
        captions_indices = self.tokenizer.texts_to_sequences(captions)
        captions_indices = [caption + [self.eos_token_index()] for caption in captions_indices]
        captions_indices = np.array(list(itertools.zip_longest(*captions_indices, fillvalue=0))).T

        max_idx = len(self.vocab) + 1

        one_hot_captions = []
        for i, caption in enumerate(captions_indices):
            one_hot_captions.append([])
            for token in caption:
                one_hot = np.zeros(max_idx)
                one_hot[token] = int(bool(token))  # padding will be array of 0s
                one_hot_captions[i].append(one_hot)

        return np.asarray(one_hot_captions)

    def decode_captions(self, one_hot_captions) -> List[str]:
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

    def word_embedding_layer(self, word_vectors: WordVector) -> List[Layer]:
        """
        Builds the embedding layer that is prepended to every RNN timestep.
        :param word_vectors: the word vector object that contains parsed
        :return:
        """
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
        data = [annotation["caption"] for annotation in data["annotations"]][:2]

        tp = TextPreprocessor()
        tp.fit_captions(data)

        wv = WordVector(tp.vocab, "fasttext")

        model = Sequential()
        [model.add(layer) for layer in tp.word_embedding_layer(wv)]

        model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))
        tmp = tp.encode_captions([TextPreprocessor.eos_token()])[0][0]
        print(model.predict(np.array([tmp])))
