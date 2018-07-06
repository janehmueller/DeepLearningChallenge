from keras import Sequential

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet
from src.rnn_net import RNNNet
from src.text_preprocessing import TextPreprocessor, word_embedding, WordVector


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def training_data(images, text_preprocessor, file_loader):
    for image_id, image in images:
        for caption in file_loader.id_caption_map[image_id]:
            yield (image, text_preprocessor.encode_caption(caption))


def main():
    file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)
    rnn_net = RNNNet()
    text_preprocessor = TextPreprocessor()
    text_preprocessor.process_captions(file_loader.id_caption_map.values())

    model = Sequential()
    model_list_add(model, image_net.layers)
    model_list_add(model, word_embedding(text_preprocessor.vocab,
                                         WordVector(text_preprocessor.vocab, "fasttext")))
    model_list_add(model, rnn_net.layers)

    model.compile(**base_configuration['model_hyper_params'])

    training_data_generator = training_data(image_net.images, text_preprocessor, file_loader)
    model.fit_generator(training_data_generator, **base_configuration['fit_params'])


if __name__ == '__main__':
    main()
