import os
import pandas as pd
from PIL import Image
import numpy as np
from scipy.spatial import distance
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns

class KNN:

    model = []
    classification = []

    def train(self, writers_real, writers_forged):

        for i in range(len(writers_real)):

        # Current method is comparing the squared differences, could possibly just compare non-squared?

            realImgs = writers_real[i]
            forgedImgs = writers_forged[i]

            print(i)

            for j in range(len(realImgs)):

                img_real = cv2.imread(realImgs[j], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                img_real = cv2.resize(img_real, (256, 128))

                # Compare to all other images in realImgs

                for k in range(j+1, len(realImgs)):

                    img_2_real = cv2.imread(realImgs[k], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    img_2_real = cv2.resize(img_2_real, (256, 128))

                    real_diff = (img_real - img_2_real)**2

                    self.model.append(real_diff)
                    self.classification.append('Real')

                # Compare to all images in forgedImgs

                for k in range(len(forgedImgs)):

                    img_forged = cv2.imread(forgedImgs[k], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    img_forged = cv2.resize(img_forged, (256, 128))

                    forged_diff = (img_real - img_forged)**2

                    self.model.append(forged_diff)
                    self.classification.append('Forged')

    def test(self, writing_1, writing_2, k):

        img_1 = cv2.imread(writing_1, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_1 = cv2.resize(img_1, (256, 128))
        img_2 = cv2.imread(writing_2, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_2 = cv2.resize(img_2, (256, 128))

        diff = (img_1 - img_2)**2

        votes = {'Real': 0, 'Forged': 0}

        for i in range(k):

            min_distance = None
            classification = None
            ind = None

            used = []

            for index in range(len(self.model)):
                row = self.model[index]
                dist = np.linalg.norm(diff - np.array(row))

                if (min_distance == None or dist < min_distance) and index not in used:
                    min_distance = dist
                    classification = self.classification[index]
                    ind = index

            votes[classification] += 1

            used.append(ind)

        if votes['Forged'] > votes['Real']:
            return('Forged')
        else:
            return('Real')

    def evaluate(self, org_list, forg_list, num_img, k, train_split, test_split):

        writers_real = [org_list[i:i + num_img] for i in range(0, len(org_list), num_img)]
        writers_forged = [forg_list[i:i + num_img] for i in range(0, len(forg_list), num_img)]

        writers_real_train = writers_real[:train_split]
        writers_forged_train = writers_forged[:train_split]

        self.train(writers_real_train, writers_forged_train)

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in range(test_split):

            print(i)

            group = random.randint(train_split, len(writers_real) - 1)
            writing_1 = writers_real[group][random.randint(0, num_img - 1)]

            same = random.choice([0,1])

            if (same):
                writing_2 = writers_real[group][random.randint(0, num_img - 1)]
            else:
                writing_2 = writers_forged[group][random.randint(0, num_img - 1)]

            result = self.test(writing_1, writing_2, k)

            if (result == "Real" and same):
                TP += 1
            elif (result == "Real" and not same):
                FP += 1
            elif (result == "Forged" and same):
                FN += 1
            elif (result == "Forged" and not same):
                TN += 1

        accuracy = (float(TP + FP) / float(TP + FP + TN + FN))
        print(accuracy)

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)

        F1_score = (2.0 * precision * recall) / (precision + recall)

        return {"F1_score": F1_score, "TP": TP, "FP": FP, "TN": TN, "FN": FN}


if __name__ == '__main__':
    IMG_DIR = './data/preprocessed_old'

    org_list = os.listdir(os.path.join(IMG_DIR, 'full_org/'))
    org_list = [os.path.join(IMG_DIR, 'full_org/' + img) for img in org_list]

    forg_list = os.listdir(os.path.join(IMG_DIR, 'full_forg'))
    forg_list = [os.path.join(IMG_DIR, 'full_forg/' + img) for img in forg_list]

    # k = 3
    train_split = 15
    test_split = 300

    knn = KNN()
    # knn.evaluate(org_list, forg_list, 24, k, train_split, test_split)

    k_values = [3,5,7,9]
    F1_scores = []

    for k_val in k_values:
        result = knn.evaluate(org_list, forg_list, 24, k_val, train_split, test_split)
        F1_scores.append(result['F1_score'])

        conf_matrix = [[result['TN'], result['FP']],
                        [result['FN'], result['TP']]]

        sns.heatmap(conf_matrix, annot=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix for k-value of: " + str(k_val))
        plt.show()

    plt.clf()

    plt.plot(k_values, F1_scores)
    plt.xlabel("K-values")
    plt.ylabel("F1 scores")
    plt.title("K-values vs. F1 score")
    plt.show()