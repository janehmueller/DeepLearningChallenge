from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence
import numpy as np
from collections import namedtuple

from src.file_loader import File
from src.text_preprocessing import TextPreprocessor
from src.config import base_configuration
from util.threading import LockedIterator
from util.functional import group

CaptionTuple = namedtuple('CaptionTuple', ['image_id', 'caption'])

class DataLoader(Sequence):
    def __init__(self, file_loader, model_dir):
        self.file_loader = file_loader
        self.text_preprocessor = TextPreprocessor()
        self.text_preprocessor.process_captions(self.file_loader.id_caption_map.values())
        self.text_preprocessor.serialize(model_dir)
        self.batch_size = base_configuration['batch_size']
        self.batch_captions = self.calc_batches()

    def calc_batches(self):
        flat_captions = [CaptionTuple(image_id, caption)
                for image_id in self.file_loader.id_file_map.keys()
                for caption in self.file_loader.id_caption_map[image_id]]
        return list(group(flat_captions, self.batch_size))

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(self.captions_num / self.batch_size))

    # get a batch
    def __getitem__(self, index):
        # TODO shuffle after every epoch
        caption_tuples = self.batch_captions[index]
        image_ids = {tup.image_id for tup in caption_tuples}
        images = {
            image_id: self.preprocess_image(self.file_loader.id_file_map[image_id]) for image_id in image_ids
        }

        image_shape = [299, 299, 3]
        caption_length = base_configuration['sizes']['repeat_vector_length']
        one_hot_size = self.text_preprocessor.one_hot_encoding_size

        batch_images = np.zeros(shape=[self.batch_size] + image_shape)
        batch_captions = np.zeros(shape=[self.batch_size, caption_length, one_hot_size])

        for i in range(self.batch_size):
            caption = caption_tuples[i].caption
            batch_captions[i] = self.text_preprocessor.encode_caption(caption)
            batch_images[i] = images[caption_tuples[i].image_id]
        return (batch_images, batch_captions)

    @staticmethod
    def preprocess_image(path):
        loaded_image = load_img(path, target_size=(299, 299))
        return img_to_array(loaded_image)

    @property
    def images_num(self):
        return len(self.file_loader.id_file_map)

    @property
    def captions_num(self):
        return sum((len(self.file_loader.id_caption_map[key]) for key in self.file_loader.id_file_map.keys()))
