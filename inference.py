import math
from argparse import ArgumentParser
from contextlib import redirect_stdout
from io import StringIO
from os import path
from typing import List, Dict

import numpy as np
from keras import Model
from keras.models import load_model

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.text_preprocessing import TextPreprocessor
from util.metrics import CIDEr, Score, BLEU, METEOR, ROUGE


def prediction_data(images):
    batch_size = base_configuration['batch_size']
    image_shape = [299, 299, 3]
    batch_images = np.zeros(shape=[batch_size] + image_shape)
    image_ids = []
    i = 0
    for image_id, image in images:
        if i >= batch_size:
            # yield (np.copy(batch_images), np.copy(batch_captions)) PROBABLY WE SHOULD USE THIS
            yield batch_images, image_ids
            i = 0
            image_ids = []

        image_ids.append(image_id)
        batch_images[i] = image
        i += 1


def predict(model: Model, data_generator, step_size, tp: TextPreprocessor) -> Dict[int, str]:
    image_id_to_prediction = {}
    for _ in range(0, 1):  # TODO: step_size
        image_batch, image_ids = next(data_generator)
        # predict first timestep only the image (and an empty caption)
        captions = predict_batch(model, [image_batch, np.zeros(shape=[base_configuration['batch_size']])], tp)
        # remove the prediction result of the empty caption
        captions = captions[:, :-1]
        i = 0
        while i < base_configuration['sizes']['repeat_vector_length']:
            captions_prediction = predict_batch(model, [image_batch, captions], tp)
            captions = captions_prediction
            last_column = captions[:, -1:]
            if np.all(last_column == tp.eos_token_index()):
                break
            i += 1

        captions_str = tp.decode_captions_to_str(captions)

        for index, image_id in enumerate(image_ids):
            image_id_to_prediction[image_id] = captions_str[index]
        # caption_results.extend(captions_str)
        # image_ids.extend(image_ids)
    return image_id_to_prediction


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

    try:
        file_loader = File.load_validation(base_configuration['selected_dataset'])
    except FileNotFoundError:
        file_loader = File.load_training(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)
    step_size = math.ceil(image_net.captions_num / base_configuration['batch_size'])

    prediction_data_generator = prediction_data(image_net.images)
    image_id_to_prediction = predict(model, prediction_data_generator, step_size, text_preprocessor)
    image_id_to_captions = {image_id: file_loader.id_caption_map[image_id] for image_id in image_id_to_prediction}

    metrics: List[Score] = [CIDEr(), BLEU(4), ROUGE()]

    print("Scores:")
    for metric in metrics:
        with redirect_stdout(StringIO()):
            scores = metric.calculate(image_id_to_prediction, image_id_to_captions)

        for name, score in scores.items():
            print("\t{}: {}".format(name, score))

    for image_id, prediction in image_id_to_prediction.items():
        print("Image path: " + file_loader.id_file_map[image_id])
        print(prediction)
        print("\t" + "\n\t".join(file_loader.id_caption_map[image_id]) + "\n")


if __name__ == '__main__':
    main()
