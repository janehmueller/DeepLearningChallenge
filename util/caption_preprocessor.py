from typing import List

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np


class CaptionPreprocessor(object):
    # End of sentence token
    EOS_TOKEN = 'zeosz'

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.word_dictionary = {}

    def eos_index(self):
        """
        Returns id of the EOS token in the vocabulary
        :return: id of the EOS token in the vocabulary
        """
        return self.tokenizer.word_index[self.EOS_TOKEN]

    def vocabs(self):
        """
        Returns word index of vocabulary sorted by the ids
        :return: word index of vocabulary sorted by the idsg
        """
        word_index = self.tokenizer.word_index
        return sorted(word_index, key=word_index.get)

    def add_eos(self, captions: List[str]):
        return map(lambda x: x + ' ' + self.EOS_TOKEN, captions)

    def preprocess_captions(self, captions: List[str]):
        # TODO: handle rare words
        captions = self.add_eos(captions)
        self.tokenizer.fit_on_texts(captions)
        self.word_dictionary = {index: word for word, index in self.tokenizer.word_index.items()}

    def encode_captions(self, captions: List[str]):
        captions = self.add_eos(captions)
        return self.tokenizer.texts_to_sequences(captions)

    def decode_captions(self, captions_prediction, expected_captions=None):
        """
        Decodes predicted captions (in one-hot-encoding) into strings.
        :param captions_prediction: captions_prediction: numpy array of one-hot encoded captions
        :param expected_captions: the captions that should be predicted
        :return: list of captions decoded as strings
        """
        captions = captions_prediction[:, :-1, :]  # Discard the one-hot encoded EOS token
        decoded_labels = captions.argmax(axis=-1)  # Returns indices of highest value in array (which is the number that was one-hot encoded)
        num_batches, num_words = decoded_labels.shape

        if expected_captions:
            captions_length = self.captions_length(expected_captions)
        else:
            captions_length = [num_words] * num_batches  # Placeholder with which we read all words

        decoded_captions = []
        for caption_index in range(0, num_batches):
            caption_string = []
            for word_index in range(0, captions_length[caption_index]):
                label = decoded_labels[caption_index, word_index]
                # label += 1
                caption_string.append(self.word_dictionary[label])
            decoded_captions.append(' '.join(caption_string))

        return decoded_captions

    def preprocess_batch(self, captions_label_encoded):
        captions = pad_sequences(captions_label_encoded, padding="post")  # pad with trailing zeros

        # The number of timesteps/words the model outputs is maxlen(captions) + 1 because the first "word" is an image
        captions_extended1 = pad_sequences(captions, maxlen=captions.shape[-1] + 1, padding="post")
        captions_one_hot = map(self.tokenizer.sequences_to_matrix, np.expand_dims(captions_extended1, -1))
        captions_one_hot = np.array(captions_one_hot, dtype="int")

        # Left-shift one-hot encoding by one to set padding to 0 (so that error will be 0.0)
        # Decrease indices to adjust for change in one-hot encoding
        captions_decreased = captions.copy()
        captions_decreased[captions_decreased > 0] -= 1
        captions_one_hot_shifted = captions_one_hot[:, :, 1:]

        captions_input = captions_decreased
        captions_output = captions_one_hot_shifted
        return captions_input, captions_output

    def normalize_captions(self, captions: List[str]):
        word_sequences = map(text_to_word_sequence, self.add_eos(captions))
        return map(' '.join, word_sequences)

    def captions_length(self, captions):
        """
        Calculates the lengths of the passed captions.
        :param captions: three-dimensional numpy array containing a batch of one-hot encoded captions
        :return: numpy array of the captions lengths (number of words in a caption)
        """
        collapsed_one_hot_encodings = captions.sum(axis=2)
        zero_filtered_captions = collapsed_one_hot_encodings != 0
        caption_lengths = zero_filtered_captions.sum(axis=1)
        return caption_lengths
