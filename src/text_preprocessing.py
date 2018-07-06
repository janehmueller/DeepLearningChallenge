import itertools
import json
from typing import List, Dict
import numpy as np
from keras.preprocessing.text import Tokenizer


class TextPreprocessor(object):

    tokenizer:Tokenizer
    eos_token:str
    dictionary:Dict[str, int]
    def __init__(self):
        self.eos_token = "zeornd"
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.eos_token])
        self._dictionary = None
        self.inverse_dictionary = None

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value
        self.inverse_dictionary = dict([(v, k) for k, v in self._dictionary.items()])

    def eos_token_index(self):
        return self.dictionary[self.eos_token]

    def process_captions(self, captions:List[str]):
        self.tokenizer.fit_on_texts(captions)
        self.dictionary = self.tokenizer.word_index

    def encode_captions(self, captions:List[str]) -> List[List[int]]:
        captions_indices = self.tokenizer.texts_to_sequences(captions)
        captions_indices = [caption + [self.eos_token_index()] for caption in captions_indices]
        captions_indices = np.array(list(itertools.zip_longest(*captions_indices, fillvalue=0))).T

        max_idx = np.max(captions_indices) + 1

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
        with open("data/vocab.json", "w") as file:
            json.dump(self.dictionary, file)

    def deserialize(self):
        with open("data/vocab.json", "r") as file:
            self.dictionary = json.load(file)
            self.tokenizer.word_index = self.dictionary


if __name__ == "__main__":

    with open("data/annotations/pretty_train.json") as file:
        data = json.load(file)
        data = [annotation["caption"] for annotation in data["annotations"]][:2]

        tp = TextPreprocessor()
        tp.process_captions(data)

        tmp = tp.encode_captions(data)
        print(tp.decode_captions(tmp))
