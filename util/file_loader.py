import json
import os

from .config import fix_for_project_root_path


class File:
    def __init__(self, path):
        self.path = fix_for_project_root_path(path)
        self.data = None

    @staticmethod
    def load(dataset_name):
        return class_dict[dataset_name]


class CocoFile(File):
    def __init__(self, path):
        super(CocoFile, self).__init__(path)
        self._id_file_map = None
        self._id_caption_map = None
        self.preprocess()

    def preprocess(self):
        with open(self.path) as file:
            self.data = json.load(file)

    @property
    def id_file_map(self):
        if not self._id_file_map:
            self._id_file_map = {}
            for annotation in self.data['images']:
                self._id_file_map[annotation['id']] = annotation['file_name']
        return self._id_file_map

    @property
    def id_caption_map(self):
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
    def __init__(self, path):
        super(Flickr30kFile, self).__init__(path)
        self._id_file_map = {}
        self._id_caption_map = {}
        self.preprocess()

    def preprocess(self):
        with open(self.path) as file:
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
    "coco": CocoFile,
    "flickr30k": Flickr30kFile
}
