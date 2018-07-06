import argparse

from util.config import base_configuration
from util.word_vectors import WordVector

import keras


def main():
    parser = argparse.ArgumentParser(description='Superawesome image captioning')
    parser.add_argument('--dataPath', dest='data_path', default=base_configuration['data_path'])

    args = parser.parse_args()

    WordVector(['one', 'two', 'three'], keras.initializers.RandomUniform(0, 1), 'glove')


if __name__ == '__main__':
    main()
