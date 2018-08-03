import math
from argparse import ArgumentParser
from os import path
from typing import List

import numpy as np
from keras import Model
from keras.models import load_model

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.text_preprocessing import TextPreprocessor

from util.loss import categorical_crossentropy_from_logits


processed_images = []


def prediction_data(images):
    batch_size = base_configuration['batch_size']
    image_shape = [299, 299, 3]
    batch_images = np.zeros(shape=[batch_size] + image_shape)
    i = 0
    for image_id, image in images:
        if i >= batch_size:
            # yield (np.copy(batch_images), np.copy(batch_captions)) PROBABLY WE SHOULD USE THIS
            yield batch_images
            i = 0

        processed_images.append(image_id)
        batch_images[i] = image
        i += 1


# def training_data(images, text_preprocessor: TextPreprocessor, file_loader: File):
#     batch_size = base_configuration['batch_size']
#     image_shape = [299, 299, 3]
#     batch_images = np.zeros(shape=[batch_size] + image_shape)
#     caption_length = base_configuration['sizes']['repeat_vector_length']
#     caption_output_length = caption_length + 1
#     one_hot_size = text_preprocessor.one_hot_encoding_size
#     batch_captions = np.zeros(shape=[batch_size, caption_output_length, one_hot_size])
#     batch_input_captions = np.zeros(shape=[batch_size, caption_length])
#     i = 0
#     for image_id, image in images:
#         for caption in file_loader.id_caption_map[image_id]:
#             if i >= batch_size:
#                 # yield (np.copy(batch_images), np.copy(batch_captions)) PROBABLY WE SHOULD USE THIS
#                 yield ([batch_images, batch_input_captions], batch_captions)
#                 i = 0
#             batch_images[i] = image
#             batch_captions[i] = text_preprocessor.encode_caption(caption)
#             batch_input_captions[i] = text_preprocessor.encode_caption(caption, one_hot=False)
#             processed_images.append(image_id)
#             batch_images[i] = image
#             i += 1


def predict(model: Model, data_generator, step_size, tp: TextPreprocessor) -> List[str]:
    caption_results = []
    for _ in range(0, 1):  # TODO: step_size
        image_batch = next(data_generator)
        captions = predict_batch(model, [image_batch, np.zeros(shape=[base_configuration['batch_size']])], tp)[:, :-1]
        i = 0
        while i < base_configuration['sizes']['repeat_vector_length']:
            captions_prediction = predict_batch(model, [image_batch, captions], tp)
            captions = captions_prediction
            last_column = captions[:, -1:]
            if np.all(last_column == tp.eos_token_index()):
                break
            i += 1
        captions_str = tp.decode_captions_to_str(captions)
        caption_results.extend(captions_str)
    return caption_results


def predict_batch(model: Model, input_batch, tp: TextPreprocessor) -> np.ndarray:
    prediction = model.predict(input_batch, batch_size=base_configuration['batch_size'])
    return tp.decode_captions_to_indices(prediction)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-name", dest="model_name", type=str)
    args = parser.parse_args()

    model_dir = path.join(base_configuration['tmp_path'], 'model-saves')
    model_path = path.join(model_dir, args.model_name + ".hdf5")
    # model_dir = '/home/cps4/DeepLearningChallenge/tmp/model-saves-leo-02'
    # model_path = path.join(model_dir, '{:02d}.hdf5'.format(model_epoch))

    text_preprocessor = TextPreprocessor()
    text_preprocessor.deserialize(model_dir)

    model = load_model(model_path, compile=False)

    file_loader = File.load(base_configuration['selected_dataset'])
    image_net = ImageNet(file_loader)
    step_size = math.ceil(image_net.captions_num / base_configuration['batch_size'])

    prediction_data_generator = prediction_data(image_net.images)
    # prediction_data_generator = training_data(image_net.images, text_preprocessor, file_loader)
    predictions = predict(model, prediction_data_generator, step_size, text_preprocessor)

    for i, prediction in enumerate(predictions):
        print(file_loader.id_file_map[processed_images[i]])
        print(prediction)
        print(file_loader.id_caption_map[processed_images[i]])

    # predictions = model.predict_generator(prediction_data_generator, steps=1)
    # captions_indices = [np.argmax(pred_caption, axis=1) for pred_caption in predictions]

    # sentences = [[text_preprocessor.inverse_vocab.get(caption, None) for caption in caption_indices] for caption_indices in captions_indices]
    # for i, s in enumerate(sentences):
    #     print(file_loader.id_file_map[processed_images[i]])
    #     print(s)
    #     print(file_loader.id_caption_map[processed_images[i]])


    # TODO translate to words via vocabulary of training pass (see serialize and deserialize in text_processing)

    # print("Shape of predictions: {}".format(predictions.shape))


if __name__ == '__main__':
    main()
