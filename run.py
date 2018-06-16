import argparse
import json
from os import path

from util.FileLoader import File


def main():
    with open(path.join(path.dirname(__file__), 'config.json')) as config_file:
        base_configurations = json.load(config_file)

    parser = argparse.ArgumentParser(description='Superawesome image captioning')
    parser.add_argument('--dataPath', dest='data_path', default=base_configurations['data_path'])

    args = parser.parse_args()

    for fileInfo in base_configurations['files']['train']:
        File(path.join(args.data_path, fileInfo['path'])).preprocess()


if __name__ == '__main__':
    main()
