from collections import Iterable
from typing import List
from functools import partial

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import inception_v3
import numpy as np


class ImagePreprocessor(object):
    IMAGE_SIZE = (299, 299)  # Inceptionv3 input size

    def preprocess_image(self, path):
        image = load_img(path, target_size=self.IMAGE_SIZE)
        image_array = img_to_array(image)
        image_array = inception_v3.preprocess_input(image_array)
        return image_array

    def preprocess_batch(self, image_list):
        return np.array(image_list)

    def preprocess_images(self, image_paths):
        # return [partial(self.preprocess_image)(path) for path in image_paths]
        return list(map(partial(self.preprocess_image), image_paths))


class CaptionPreprocessor(object):
    # End of sentence token
    EOS_TOKEN = 'zeosz'

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.word_dictionary = {}
        self.word_of = None

    @property
    def EOS_TOKEN_LABEL_ENCODED(self):
        return self.tokenizer.word_index[self.EOS_TOKEN]

    def eos_index(self):
        """
        Returns id of the EOS token in the vocabulary
        :return: id of the EOS token in the vocabulary
        """
        return self.tokenizer.word_index[self.EOS_TOKEN]

    def vocabs(self):
        """
        Returns word index of vocabulary sorted by the ids
        :return: word index of vocabulary sorted by the ids
        """
        word_index = self.tokenizer.word_index
        return sorted(word_index, key=word_index.get)

    @property
    def vocab_size(self):
        return len(self.tokenizer.word_index)

    def add_eos(self, captions: Iterable):
        # return (caption + ' ' + self.EOS_TOKEN for caption in captions)
        return map(lambda x: x + ' ' + self.EOS_TOKEN, captions)

    def preprocess_captions(self, captions: List[str]):
        # TODO: handle rare words
        captions = self.add_eos(captions)
        self.tokenizer.fit_on_texts(captions)
        self.word_dictionary = {index: word for word, index in self.tokenizer.word_index.items()}

    def encode_captions(self, captions: Iterable):
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

        if expected_captions is not None:
            captions_length = self.captions_length(expected_captions)
        else:
            captions_length = [num_words] * num_batches  # Placeholder with which we read all words

        decoded_captions = []
        for caption_index in range(0, num_batches):
            caption_string = []
            for word_index in range(0, captions_length[caption_index]):
                label = decoded_labels[caption_index, word_index]
                # print(decoded_labels)
                # print("WORDS")
                # print("\n".join(self.word_dictionary))
                label += 1
                # caption_string.append(self.word_dictionary[label])
                caption_string.append(self.word_dictionary.get(label, "BROKEN"))
            decoded_captions.append(' '.join(caption_string))

        return decoded_captions

    def preprocess_batch(self, captions_label_encoded):
        captions = pad_sequences(captions_label_encoded, padding="post")  # pad with trailing zeros

        # The number of timesteps/words the model outputs is maxlen(captions) + 1 because the first "word" is an image
        captions_extended1 = pad_sequences(captions, maxlen=captions.shape[-1] + 1, padding="post")
        # captions_one_hot = [self.tokenizer.sequences_to_matrix(seq) for seq in np.expand_dims(captions_extended1, -1)]
        captions_one_hot = list(map(self.tokenizer.sequences_to_matrix, np.expand_dims(captions_extended1, -1)))
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
        # word_sequences = (text_to_word_sequence(elem) for elem in self.add_eos(captions))
        word_sequences = map(text_to_word_sequence, self.add_eos(captions))
        # return (' '.join(caption) for caption in word_sequences)
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

    def fit_on_captions(self, captions_txt):
        # captions_txt = self.handle_rare_words(captions_txt)
        captions_txt = self.add_eos(captions_txt)
        self.tokenizer.fit_on_texts(captions_txt)
        self.word_of = {i: w for w, i in self.tokenizer.word_index.items()}
