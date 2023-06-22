import numpy as np
import os
import cv2
import random
import pickle

DATA = ['training_data', 'testing_data']
CATEGORIES = ['face', 'spaceship']
IMG_SIZE = 32


def create_training_data(folder):
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(folder, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print("Error:", e)
                pass
    return training_data

for folder in DATA:

    data = create_training_data(folder)
    random.shuffle(data)

    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    # Normalize features
    features = features / 255.0

    pickle_out = open(folder + "/features.pickle", "wb")
    pickle.dump(features, pickle_out)
    pickle_out.close()

    pickle_out = open(folder + "/labels.pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()