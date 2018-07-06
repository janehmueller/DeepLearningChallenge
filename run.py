import argparse

from util.config import base_configuration

import keras


def main():
    parser = argparse.ArgumentParser(description='Superawesome image captioning')
    parser.add_argument('--dataPath', dest='data_path', default=base_configuration['data_path'])

    args = parser.parse_args()

    print(base_configuration)


if __name__ == '__main__':
    main()
