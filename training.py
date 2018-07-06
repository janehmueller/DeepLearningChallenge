from keras import Sequential

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.rnn_net import RNNNet
from src.text_preprocessing import TextPreprocessor, word_embedding, WordVector


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def main():
    file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)
    rnn_net = RNNNet()
    text_preprocessor = TextPreprocessor()
    picture_id_caption_one_hot = text_preprocessor.process_id_to_captions(file_loader.id_caption_map)

    model = Sequential()
    model_list_add(model, image_net.layers)
    model_list_add(model, word_embedding(text_preprocessor.dictionary,
                                         WordVector(text_preprocessor.dictionary, "fasttext")))
    model_list_add(model, rnn_net.layers)

    model.compile(**base_configuration['model_hyper_params'])

    training_data_generator = ((image, picture_id_caption_one_hot[image_id]) for (image_id, image) in image_net.images)

    model.fit_generator(training_data_generator, **base_configuration['fit_params'])


if __name__ == '__main__':
    main()
