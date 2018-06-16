import argparse
import json
from os import path

from util.file_loader import File, TrainFile


def main():
    with open(path.join(path.dirname(__file__), 'config.json')) as config_file:
        base_configurations = json.load(config_file)

    parser = argparse.ArgumentParser(description='Superawesome image captioning')
    parser.add_argument('--dataPath', dest='data_path', default=base_configurations['data_path'])

    args = parser.parse_args()

    file_data = TrainFile(path.join(args.data_path, base_configurations['files']['train']['captions']['path']))

    print(file_data)


if __name__ == '__main__':
    main()
