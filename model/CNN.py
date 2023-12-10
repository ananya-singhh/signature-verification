import os
import numpy as np
from scipy.spatial import distance
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from keras import datasets
from keras.models import Sequential
from keras import layers
from keras import regularizers

class CNN:

    train_data = []
    train_labels = []
    width = None
    height = None

    model = None

    def get_train_data(self, writers_real, writers_forged):

        for i in range(len(writers_real)):

        # Current method is comparing the squared differences, could possibly just compare non-squared?

            realImgs = writers_real[i]
            forgedImgs = writers_forged[i]

            self.width = 256
            self.height = 128

            for j in range(len(realImgs)):

                img_real = cv2.imread(realImgs[j], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                img_real = cv2.resize(img_real, (256, 128))

                # Compare to all other images in realImgs

                for k in range(j+1, len(realImgs)):

                    img_2_real = cv2.imread(realImgs[k], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    img_2_real = cv2.resize(img_2_real, (256, 128))

                    real_diff = (img_real - img_2_real)**2

                    self.train_data.append(real_diff)
                    self.train_labels.append(0)

                # Compare to all images in forgedImgs

                for k in range(len(forgedImgs)):

                    img_forged = cv2.imread(forgedImgs[k], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    img_forged = cv2.resize(img_forged, (256, 128))

                    forged_diff = (img_real - img_forged)**2

                    self.train_data.append(forged_diff)
                    self.train_labels.append(1)

    def train(self, test_images, test_labels):

        class_names = ['Real', 'Forged']

        model = keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_regularizer=regularizers.l2(0.01),
                                input_shape=(self.height, self.width, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Dropout(0.5))

        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2))

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        results = model.fit(np.array(self.train_data), np.array(self.train_labels), epochs=3,
                            validation_data=(np.array(test_images), np.array(test_labels)))

        self.model = model

    def test(self, test_data, test_labels):

        test_loss, test_accuracy = self.model.evaluate(test_data, test_labels, verbose=1)

        print(test_loss)
        print(test_accuracy)

    def evaluate(self, org_list, forg_list, num_img, train_split, test_split):

        writers_real = [org_list[i:i + num_img] for i in range(0, len(org_list), num_img)]
        writers_forged = [forg_list[i:i + num_img] for i in range(0, len(forg_list), num_img)]

        writers_real_train = writers_real[:train_split]
        writers_forged_train = writers_forged[:train_split]

        self.get_train_data(writers_real_train, writers_forged_train)

        test_data = []
        test_labels = []

        for i in range(test_split):

            group = random.randint(train_split, len(writers_real) - 1)
            writing_1 = writers_real[group][random.randint(0, num_img - 1)]

            same = random.choice([0,1])

            if (same):
                writing_2 = writers_real[group][random.randint(0, num_img - 1)]
                test_labels.append(0)
            else:
                writing_2 = writers_forged[group][random.randint(0, num_img - 1)]
                test_labels.append(1)

            img_1 = cv2.imread(writing_1, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            img_1 = cv2.resize(img_1, (256, 128))
            img_2 = cv2.imread(writing_2, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            img_2 = cv2.resize(img_2, (256, 128))

            diff = (img_1 - img_2)**2

            test_data.append(diff)

        self.train(test_data, test_labels)
        # self.test(test_data, test_labels)

if __name__ == '__main__':
    IMG_DIR = './data/preprocessed'

    org_list = os.listdir(os.path.join(IMG_DIR, 'full_org/'))
    org_list = [os.path.join(IMG_DIR, 'full_org/' + img) for img in org_list]

    forg_list = os.listdir(os.path.join(IMG_DIR, 'full_forg'))
    forg_list = [os.path.join(IMG_DIR, 'full_forg/' + img) for img in forg_list]

    train_split = 15
    test_split = 1000

    cnn = CNN()
    cnn.evaluate(org_list, forg_list, 24, train_split, test_split)