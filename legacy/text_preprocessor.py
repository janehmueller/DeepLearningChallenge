import itertools
import numpy as np
from itertools import groupby
from typing import List, Dict

from keras.preprocessing.text import Tokenizer

from src.config import base_configuration


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
        flat_id_to_captions = [(image_id, caption) for image_id, image_captions in id_to_captions.items() for caption in
                               image_captions]
        return zip(*list(flat_id_to_captions))

    def zip_flat_id_to_captions(self, ids: List[int], encoded_captions: List[np.ndarray]) -> Dict[int, List[np.ndarray]]:
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

    def process_id_to_captions(self, id_to_captions: Dict[int, List[str]]) -> Dict[int, List[np.ndarray]]:
        ids, captions = self.unzip_and_flatten_id_to_captions(id_to_captions)
        encoded_captions = self.process_captions(captions)
        return self.zip_flat_id_to_captions(ids, encoded_captions)

    def fit_on_id_to_captions(self, id_to_captions: Dict[int, List[str]]):
        ids, captions = self.unzip_and_flatten_id_to_captions(id_to_captions)
        self.fit_captions(captions)

    def encode_id_to_captions(self, id_to_captions: Dict[int, List[str]]) -> Dict[int, List[np.ndarray]]:
        ids, captions = self.unzip_and_flatten_id_to_captions(id_to_captions)
        encoded_captions = self.encode_captions(captions)
        return self.zip_flat_id_to_captions(ids, encoded_captions)

    def process_captions(self, captions: List[str]) -> List[np.ndarray]:
        """
        Updates the vocabulary with the captions and encodes the captions.
        :param captions: list of captions as string
        :return: the one-hot encoded captions as numpy array
        """
        self.fit_captions(captions)
        return self.encode_captions(captions)

    def fit_captions(self, captions: List[str]):
        """
        Updates the vocabulary with the captions.
        :param captions: list of captions as string
        """
        self.tokenizer.fit_on_texts(captions)
        self.vocab = self.tokenizer.word_index

    def one_hot_encode_caption(self, caption_indices: List[int], one_hot_size: int) -> np.ndarray:
        one_hot = np.zeros([len(caption_indices), one_hot_size])
        one_hot[np.arange(len(caption_indices)), caption_indices] = 1
        return np.pad(one_hot[:, 1:], [1, 0], mode='constant', constant_values=0)[1:]

    def encode_captions(self, captions: List[str]) -> List[np.ndarray]:
        """
        Tokenizes and one-hot encodes captions. They are returned as numpy array with the shape
        (num_captions, size_of_longest_caption, vocab_size + 1). Padding is encoded as zero-vector.
        :param captions: list of the captions as string
        :return: list of the one-hot encoded captions as numpy arrays
        """
        captions_indices = self.tokenizer.texts_to_sequences(captions)
        captions_indices = [caption + [self.eos_token_index()] for caption in captions_indices]
        captions_indices = np.array(list(itertools.zip_longest(*captions_indices, fillvalue=0))).T

        max_idx = len(self.vocab) + 1
        return [self.one_hot_encode_caption(caption, max_idx) for caption in captions_indices]
