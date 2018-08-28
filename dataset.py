import cv2
import os
import numpy as np
import random
import pickle

TRAINING_DIR = "fruits-360/Training/"
TEST_DIR = "fruits-360/Test/"
CLASSES = ["Banana", "Lemon", "Mango", "Orange", "Pineapple", "Guava", "Carambula", "Cocos", "Granadilla", "Kiwi"]

def image_to_array(imagePath):
    #Convert an image path to an array that can be used for classification
    return cv2.imread(imagePath)

def get_label(label, one_hot=False):
    if not one_hot:
        return CLASSES.index(label)
    vector = np.zeros(len(CLASSES))
    vector[(CLASSES.index(label))] = 1
    return vector

def get_training_data():
    try:
        training_pickle = open(os.path.join(TRAINING_DIR, "training.pkl"), 'rb')
        images, labels = pickle.load(training_pickle)
        #print("Loaded training_data from pickle")
        training_pickle.close()

        combo = list(zip(images, labels))
        random.shuffle(combo)
        images, labels = zip(*combo)

        return images, labels
    except FileNotFoundError:
        #print("Constructing training data from scratch")
        raw_data = []
        #load the images into a single array
        for subdir, dirs, files in os.walk(TRAINING_DIR):
            for file in files:
                label = os.path.basename(subdir)
                if label in CLASSES:
                    path = os.path.join(subdir, file)
                    raw_data.append((image_to_array(path), get_label(label)))

        #Break it into a training set and validation set
        random.shuffle(raw_data)
        split_index = len(raw_data) // 5
        training_data = raw_data[split_index:]
        validation_data = raw_data[:split_index]

        training_images, training_labels = format_data(training_data)
        validation_images, validation_labels = format_data(validation_data)

        #Save each data set
        training_pickle = open(os.path.join(TRAINING_DIR, "training.pkl"), 'wb')
        validation_pickle = open(os.path.join(TRAINING_DIR, "validation.pkl"), 'wb')

        pickle.dump((training_images, training_labels), training_pickle)
        pickle.dump((validation_images, validation_labels), validation_pickle)

        training_pickle.close()
        validation_pickle.close()

        return training_images, training_labels

def get_test_data(batch_size):
    data = []
    for subdir, dirs, files in os.walk(TRAINING_DIR):
        for file in files:
            label = os.path.basename(subdir)
            if label in CLASSES:
                path = os.path.join(subdir, file)
                data.append((image_to_array(path), get_label(label)))

    test_images, test_labels = format_data(data)
    iteration = 1
    while batch_size * iteration < len(data):
        iteration += 1

    return test_images[:batch_size * (iteration - 1)], test_labels[:batch_size * (iteration - 1)]

def get_validation_data(batch_size):
    validation_pickle = open(os.path.join(TRAINING_DIR, "validation.pkl"), 'rb')
    images, labels = pickle.load(validation_pickle)
    #print("Loaded validation data from pickle")
    validation_pickle.close()
    iteration = 1
    while batch_size * iteration < len(images):
        iteration += 1

    return images[:batch_size * (iteration - 1)], labels[:batch_size * (iteration - 1)]

def format_data(data):
    random.shuffle(data)
    #Gets one batch of training data
    image_data = np.zeros((len(data), 100, 100, 3), dtype=np.float64)
    image_labels = np.zeros((len(data)))
    for index, image_label_pair in enumerate(data):
        image_data[index] = image_label_pair[0]
        image_labels[index] = image_label_pair[1]
    return image_data, image_labels
