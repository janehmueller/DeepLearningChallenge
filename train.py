from keras.preprocessing.image import img_to_array, load_img
from keras.applications import inception_v3


def preprocess_image(path):
    image_size = (299, 299)
    image = load_img(path, target_size=image_size)
    image_array = img_to_array(image)
    image_array = inception_v3.preprocess_input(image_array)
    return image_array


if __name__ == "__main__":
    print("test")
