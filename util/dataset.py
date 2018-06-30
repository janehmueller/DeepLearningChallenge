import os
from collections import namedtuple

from .config import base_configuration, fix_for_project_root_path
from .file_loader import File

Datum = namedtuple('Datum', 'img_filename img_path caption_txt all_captions_txt')


class Dataset(object):
    def __init__(self, dataset_name, single_caption=False):
        self._DATASET_DIR_NAME = fix_for_project_root_path(base_configuration['data_path'])
        self._TRAINING_RESULTS_DIR_NAME = fix_for_project_root_path(base_configuration['tmp_path'])
        self._single_caption = single_caption
        self._create_dirs()

        config_dict = base_configuration['datasets'][dataset_name]
        img_train_dir_name = config_dict['train']['train_dir']
        annotation_train_file = config_dict['train']['train_annotation_file']
        img_validation_dirname = config_dict['validation']['validation_dir']
        annotation_validation_file = config_dict['validation']['validation_annotation_file']

        self._IMG_TRAIN_DIRNAME = os.path.join(self._DATASET_DIR_NAME, img_train_dir_name)
        self._ANNOTATION_TRAIN_FILE = os.path.join(self._DATASET_DIR_NAME, annotation_train_file)
        self._IMG_VALIDATION_DIRNAME = os.path.join(self._DATASET_DIR_NAME, img_validation_dirname)
        self._ANNOTATION_VALIDATION_FILE = os.path.join(self._DATASET_DIR_NAME, annotation_validation_file)

        self.FileClass = File.load(dataset_name)

        self._build()

    @property
    def training_set(self):
        return self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    @property
    def training_set_size(self):
        return len(self._training_set)

    @property
    def validation_set_size(self):
        return len(self._validation_set)

    @property
    def dataset_dir(self):
        return self._DATASET_DIR_NAME

    @property
    def training_results_dir(self):
        return self._TRAINING_RESULTS_DIR_NAME

    def _create_dirs(self):
        os.makedirs(self.training_results_dir, exist_ok=True)

    def _build(self):
        self._train_file = self.FileClass(self._ANNOTATION_TRAIN_FILE)
        self._training_set = self._build_set(self._IMG_TRAIN_DIRNAME, self._train_file)
        self._validation_file = self.FileClass(self._ANNOTATION_VALIDATION_FILE)
        self._validation_set = self._build_set(self._IMG_VALIDATION_DIRNAME, self._validation_file)

    def _build_set(self, img_dir_name, train_file, max_files=10):
        dataset = []
        count = 0
        for imageId, image_file_name in train_file.id_file_map.items():
            for caption in train_file.id_caption_map[imageId]:
                dataset.append(Datum(img_filename=image_file_name,
                                     img_path=os.path.join(img_dir_name, image_file_name),
                                     caption_txt=caption,
                                     all_captions_txt=train_file.id_caption_map[imageId]))
                if self._single_caption:
                    break

            count += 1
            if max_files and count >= max_files:
                break

        return dataset
