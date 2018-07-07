from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence
import numpy as np

from src.file_loader import File
from src.text_preprocessing import TextPreprocessor
from src.config import base_configuration
from util.threading import LockedIterator


class DataLoader(Sequence):
    def __init__(self, file_loader, model_dir):
        self.file_loader = file_loader
        self.text_preprocessor = TextPreprocessor()
        self.text_preprocessor.process_captions(self.file_loader.id_caption_map.values())
        self.text_preprocessor.serialize(model_dir)
        self.batch_size = base_configuration['batch_size']
        self.generator = self.training_data(self.images, self.text_preprocessor, self.file_loader)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.captions_num / self.batch_size))

    def __getitem__(self, index):
        return next(self.generator)

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

    @property
    def images(self):
        def generator():
            while True:
                print('RESTARTING IMAGE GENERATOR')
                for file_id, path in self.file_loader.id_file_map.items():
                    yield (file_id, self.preprocess_image(path))

        return LockedIterator(generator())


    def training_data(self, images, text_preprocessor: TextPreprocessor, file_loader: File):
        image_shape = [299, 299, 3]
        batch_images = np.zeros(shape=[self.batch_size] + image_shape)
        caption_length = base_configuration['sizes']['repeat_vector_length']
        one_hot_size = text_preprocessor.one_hot_encoding_size
        batch_captions = np.zeros(shape=[self.batch_size, caption_length, one_hot_size])
        i = 0
        for image_id, image in images:
            for caption in self.file_loader.id_caption_map[image_id]:
                if i >= self.batch_size:
                    # yield (np.copy(batch_images), np.copy(batch_captions)) PROBABLY WE SHOULD USE THIS
                    yield (batch_images, batch_captions)
                    i = 0
                batch_images[i] = image
                batch_captions[i] = text_preprocessor.encode_caption(caption)
                i += 1
