import itertools
import json
from os import path
from typing import List, Dict

import numpy as np
from keras import Sequential
from keras.engine import Layer
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer

from src.config import base_configuration
from src.file_loader import File
from src.word_vector import WordVector


class TextPreprocessor(object):
    """
    Preprocessor that tokenizes and one-hot encodes captions.
    """
    VOCAB_FILE: str = "vocab.json"
    tokenizer: Tokenizer = None
    _vocab: Dict[str, int] = None
    inverse_vocab: Dict[int, str] = None
    word_vectors: WordVector = None
    word_vector_type: str = None

    def __init__(self, word_vector_type: str = "fasttext"):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.eos_token()])
        self.word_vector_type = word_vector_type

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

    @property
    def one_hot_encoding_size(self):
        """
        Add 1 since the vocab does not contain 0 as index. Therefore a vocab with size 1 needs a vector with the shape
        (2 = vocab_size + 1) for its one-hot encoding.
        :return:
        """
        return self.vocab_size() + 1

    def eos_token_index(self) -> int:
        """
        Returns the index of the end-of-string token in the vocabulary
        :return: the index as int
        """
        return self.vocab[self.eos_token()]

    def process_captions(self, captions: List[List[str]]):
        flat_captions = list(itertools.chain(*captions))

        self.tokenizer.fit_on_texts(flat_captions)
        self.vocab = self.tokenizer.word_index

    def encode_caption(self, caption):
        return self.encode_captions([caption])[0]

    def one_hot_encode_caption(self, caption_indices: List[int], one_hot_size: int) -> np.ndarray:
        one_hot = np.zeros([len(caption_indices), one_hot_size])

        for index, value in enumerate(caption_indices):
            one_hot[index][value] = bool(value)

        # one_hot[np.arange(len(caption_indices)), caption_indices] = 1
        # Transform padding one-hot encoding with a 0-filled vector
        return one_hot

    def encode_captions(self, captions: List[str]) -> List[np.ndarray]:
        """
        Tokenizes and one-hot encodes captions. They are returned as numpy array with the shape
        (num_captions, size_of_longest_caption, vocab_size + 1). Padding is encoded as zero-vector.
        :param captions: list of the captions as string
        :return: list of the one-hot encoded captions as numpy arrays
        """
        captions_indices = self.tokenizer.texts_to_sequences(captions)
        captions_indices = [caption + [self.eos_token_index()] for caption in captions_indices]

        # TODO refactor to np.pad!
        captions_indices.append([0] * base_configuration['sizes']['repeat_vector_length'])
        captions_indices = np.array(list(itertools.zip_longest(*captions_indices, fillvalue=0))).T
        captions_indices = captions_indices[:-1]

        max_idx = self.one_hot_encoding_size
        return [self.one_hot_encode_caption(caption, max_idx) for caption in captions_indices]

    def decode_caption(self, one_hot_caption):
        return self.decode_captions(np.asarray([one_hot_caption]))[0]

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

    def serialize(self, store_path):
        with open(path.join(store_path, self.VOCAB_FILE), "w") as file:
            json.dump(self.vocab, file)

    def deserialize(self, store_path):
        with open(path.join(store_path, self.VOCAB_FILE), "r") as file:
            self.vocab = json.load(file)
            self.tokenizer.word_index = self.vocab

    def word_embedding_layer(self) -> List[Layer]:
        """
        Builds the embedding layer that is prepended to every RNN timestep.
        :return: one-element list of the dense layer
        """

        if not self.word_vectors:
            self.word_vectors = WordVector(self.vocab, self.word_vector_type)
        output_size = self.word_vectors.embedding_size()
        input_size = self.one_hot_encoding_size

        sorted_vocab = list(self.vocab.items())
        sorted_vocab.sort(key=lambda x: x[1])

        word_vector_weights = []
        word_vector_weights.append(np.zeros(output_size))
        for caption, idx in sorted_vocab:
            caption_word_vector = self.word_vectors.vectorize_word(caption)

            if caption_word_vector is None:
                caption_word_vector = np.random.normal(size=output_size, scale=np.sqrt(2. / (output_size + input_size)))

            word_vector_weights.append(caption_word_vector)

        biases = np.zeros(output_size)

        layer = Dense(
            output_size,
            input_shape=[input_size],
            weights=[np.asarray(word_vector_weights), biases],
            trainable=False
        )
        return [layer]


if __name__ == "__main__":
    file_loader = File.load(base_configuration['selected_dataset'])
    captions = file_loader.captions()[:10]
    tp = TextPreprocessor()
    tp.process_captions([captions])
    [print(tp.decode_caption(tp.encode_caption(cap)) + cap) for cap in captions[:10]]

        # model = Sequential()
        # [model.add(layer) for layer in tp.word_embedding_layer()]
        #
        # model.compile(loss="mean_squared_error", optimizer=SGD(lr=1e-4))
        # tmp = tp.encode_captions([TextPreprocessor.eos_token()])[0][0]
        # print(model.predict(np.array([tmp])))
