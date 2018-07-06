from keras import Sequential

from src.config import base_configuration
from src.file_loader import File
from src.image_net import ImageNet


def model_list_add(model: Sequential, layer_list):
    for new_layer in layer_list:
        model.add(new_layer)


def main():
    file_loader = File.load(base_configuration['selected_dataset'])

    image_net = ImageNet(file_loader)

    model = Sequential()
    model_list_add(model, image_net.layers)

    model.compile(**base_configuration['model_hyper_params'])

    for image in image_net.images:
        print(image)


if __name__ == '__main__':
    main()
