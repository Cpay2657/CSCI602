import os
# from keras.preprocessing import image
import keras.utils as image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(data_path, input_shape):
    images = []
    labels = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            _, class_label = os.path.split(os.path.normpath(root))
            image_path = os.path.join(root, file)
            img = image.load_img(image_path, target_size=input_shape)
            img_arr = image.img_to_array(img, dtype='float')
            norm_img_arr = img_arr / 255
            images.append(norm_img_arr)
            labels.append(class_label)
    label_encoder = LabelBinarizer()
    labels = label_encoder.fit_transform(labels)
    return images, labels

def preprocess_data(img_data, class_labels):
    x, x_test, y, y_test = train_test_split(img_data,
                                            class_labels,
                                            shuffle=True,
                                            stratify=class_labels,
                                            test_size=.2)

    class_encoder = LabelBinarizer()
    y = class_encoder.fit_transform(y)
    y_test = class_encoder.fit_transform(y_test)
    return np.array(x), np.array(y), np.array(x_test), np.array(y_test)

if __name__ == '__main__':
    d = load_data(data_path="/home/phat/PycharmProjects/DiffusionGenarativeModel/train_data/amazon", input_shape=(128, 128, 3))