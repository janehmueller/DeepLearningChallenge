import json
import os

from util.config import fix_for_project_root_path


class File:
    def __init__(self, path):
        self.path = fix_for_project_root_path(path)
        self.data = None
        self.preprocess()

    def preprocess(self):
        with open(self.path) as file:
            self.data = json.load(file)


class TrainFile(File):
    def __init__(self, path):
        super(TrainFile, self).__init__(path)
        self._id_file_map = None
        self._id_caption_map = None

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
