import json
from os import path

import abc
from typing import Dict, List

from .config import base_configuration


class File:
    DATASET_NAME = None

    def __init__(self, dataset):
        self.data = None
        self.dataset = dataset

    @staticmethod
    def load_training(dataset_name):
        return class_dict[dataset_name]('train')

    @staticmethod
    def load_validation(dataset_name):
        return class_dict[dataset_name]('validation')

    @property
    def annotation_path(self):
        return path.join(base_configuration['data_path'],
                         base_configuration['datasets'][self.DATASET_NAME][self.dataset]['annotation_file'])

    @property
    @abc.abstractmethod
    def id_file_map(self):
        pass

    @property
    @abc.abstractmethod
    def id_caption_map(self):
        pass

    @property
    def image_base_path(self):
        return path.join(base_configuration['data_path'],
                         base_configuration['datasets'][self.DATASET_NAME][self.dataset]['image_dir'])


class CocoFile(File):
    DATASET_NAME = 'coco'

    def __init__(self, dataset):
        super(CocoFile, self).__init__(dataset)
        self._id_file_map: Dict[str, int] = None
        self._id_caption_map: Dict[int, List[str]] = None
        self.preprocess()

    def preprocess(self):
        with open(self.annotation_path) as file:
            self.data = json.load(file)

            if base_configuration['image_input_num']:
                self.data['images'] = self.data['images'][:base_configuration['image_input_num']]

    @property
    def id_file_map(self) -> Dict[str, int]:
        if not self._id_file_map:
            self._id_file_map = {}
            for annotation in self.data['images']:
                self._id_file_map[annotation['id']] = path.join(self.image_base_path, annotation['file_name'])

        return self._id_file_map

    @property
    def id_caption_map(self) -> Dict[int, List[str]]:
        if not self._id_caption_map:
            self._id_caption_map = {}
            for annotation in self.data['annotations']:
                if annotation['image_id'] not in self._id_caption_map:
                    self._id_caption_map[annotation['image_id']] = []
                self._id_caption_map[annotation['image_id']].append(annotation['caption'])

        return self._id_caption_map

    def captions(self):
        return [annotation['caption'] for annotation in self.data['annotations']]


class Flickr30kFile(File):
    DATASET_NAME = 'flickr30k'

    def __init__(self, dataset):
        super(Flickr30kFile, self).__init__(dataset)
        self._id_file_map: Dict[str, int] = {}
        self._id_caption_map: Dict[int, List[str]] = {}
        self.preprocess()

    def preprocess(self):
        with open(self.annotation_path) as file:
            for line in file:
                tokens = line.split("/t")
                image_name, caption_nr = tokens[0].split("#")
                image_id = image_name.split(".")[0]
                annotation = tokens[1]

                self._id_file_map[image_id] = image_name
                if not image_id in self._id_caption_map:
                    self.id_caption_map[image_id] = []
                self._id_caption_map[image_id].append(annotation)

    @property
    def id_file_map(self):
        return self._id_file_map

    @property
    def id_caption_map(self):
        return self._id_caption_map


class_dict = {
    CocoFile.DATASET_NAME: CocoFile,
    Flickr30kFile.DATASET_NAME: Flickr30kFile,
}
