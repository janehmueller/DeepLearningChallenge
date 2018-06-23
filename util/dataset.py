import os
from collections import namedtuple

from .config import base_configuration, fix_for_project_root_path
from .file_loader import TrainFile

Datum = namedtuple('Datum', 'img_filename img_path '
                            'caption_txt all_captions_txt')


class Dataset(object):
    def __init__(self, dataset_name, single_caption):
        self._DATASET_DIR_NAME = fix_for_project_root_path(base_configuration['data_path'])
        self._TRAINING_RESULTS_DIR_NAME = fix_for_project_root_path(base_configuration['tmp_path'])
        self._single_caption = single_caption
        self._create_dirs()

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


class CocoDataset(Dataset):
    DATASET_NAME = 'coco'

    def __init__(self, single_caption=False):
        super(CocoDataset, self).__init__(self.DATASET_NAME, single_caption)
        self._IMG_TRAIN_DIRNAME = os.path.join(self._DATASET_DIR_NAME, 'train2014')
        self._ANNOTATION_TRAIN_FILE = os.path.join(self._DATASET_DIR_NAME, 'annotations/captions_train2014.json')
        self._IMG_VALIDATION_DIRNAME = os.path.join(self._DATASET_DIR_NAME, 'val2014')
        self._ANNOTATION_VALIDATION_FILE = os.path.join(self._DATASET_DIR_NAME, 'annotations/captions_val2014.json')

        self._build()

    def _build(self):
        self._train_file = TrainFile(self._ANNOTATION_TRAIN_FILE)
        self._training_set = self._build_set(self._IMG_TRAIN_DIRNAME, self._train_file)
        self._validation_file = TrainFile(self._ANNOTATION_VALIDATION_FILE)
        self._validation_set = self._build_set(self._IMG_VALIDATION_DIRNAME, self._validation_file)

    def _build_set(self, img_dir_name, train_file):
        dataset = []
        for imageId, image_file_name in train_file.id_file_map.items():
            for caption in train_file.id_caption_map[imageId]:
                dataset.append(Datum(img_filename=image_file_name,
                                     img_path=os.path.join(img_dir_name, image_file_name),
                                     caption_txt=caption,
                                     all_captions_txt=train_file.id_caption_map[imageId]))
                if self._single_caption:
                    break

        return dataset
