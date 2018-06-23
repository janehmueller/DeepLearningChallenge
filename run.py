import argparse
from os import path

from util import training
from util.config import base_configuration
from util.word_vectors import WordVector

import keras


def main():
    parser = argparse.ArgumentParser(description='Superawesome image captioning')
    parser.add_argument('--dataPath', dest='data_path', default=base_configuration['data_path'])

    args = parser.parse_args()

    WordVector(['one', 'two', 'three'], keras.initializers.RandomUniform(0, 1), 'glove')


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_label")
    parser.add_argument("--training_dir")
    parser.add_argument("--load_model_weights", action="store_true")
    parser.add_argument("--log_metrics_period", default=4)
    parser.add_argument("--unit_test", action="store_true")
    args = parser.parse_args()
    training.main(args.training_label,
         training_dir=args.training_dir,
         load_model_weights=args.load_model_weights,
         log_metrics_period=args.log_metrics_period,
         unit_test=args.unit_test)
