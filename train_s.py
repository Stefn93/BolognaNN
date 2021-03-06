import os
import random as rand
import numpy as np
import re
from PIL import Image
from keras.layers import *
from keras.models import *
from datetime import datetime
import time

datasetDir = 'bologna_dataset_sparse/'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'
augmented_trainsetDir = 'bologna_augmented_train_sparse/'
augmented_testsetDir = 'bologna_augmented_test_sparse/'
__train__ = trainsetDir
__test__ = testsetDir

batchSize = 128
epochs = 200
num_classes = 2
train_examples_num = len(os.listdir(__train__))
test_examples_num = len(os.listdir(__test__))


# Data augmentation, extract 2 images from each image and saves it into new folder
def data_augmentation_doublecrop(dir, newDir, num_samples):
    files = os.listdir(dir)
    i = 1
    for image_name in files:
        image = Image.open(dir + image_name)
        first_crop = (image.crop((50, 50, 250, 250))).resize((300, 300), Image.ANTIALIAS)
        second_crop = (image.crop((100, 100, 200, 200))).resize((300, 300), Image.ANTIALIAS)
        modified_name1 = (image_name.replace(".jpg", "")) + '-aug1' + '.jpg'
        modified_name2 = (image_name.replace(".jpg", "")) + '-aug2' + '.jpg'
        image.save(newDir + image_name)
        first_crop.save(newDir + modified_name1)
        second_crop.save(newDir + modified_name2)

        print("Image " + str(i) + "/" + str(num_samples))
        i += 1


def data_augmentation_0_180(dir, newDir, num_samples):
    files = os.listdir(dir)
    i = 1
    for image_name in files:
        image = Image.open(dir + image_name)
        if "_0deg" in image_name or "_180deg" in image_name:
            crop = (image.crop((100, 100, 200, 200))).resize((300, 300), Image.ANTIALIAS)
            new_name = image_name.replace(".jpg", "-aug.jpg")
            crop.save(newDir + new_name)
        image.save(newDir + image_name)

        print("Image " + str(i) + "/" + str(num_samples))
        i += 1

# Shuffle the images in the folder and returns list of shuffled image names
def shuffleList(folderPath):
    files = os.listdir(folderPath)
    rand.shuffle(files)
    return files

# Extract x,y coordinates from image name
def getLabel(image_name):
    image_name = image_name.replace(".jpg", "")
    image_name = image_name.split('_')
    x = float(image_name[0].replace('X', ""))
    y = float(image_name[1].replace('Y', ""))
    return [x, y]

def getAltLabel(image_name):
    image_name = image_name.replace(".jpg", "")
    image_name = image_name.split('_')
    x = float(image_name[1].replace('X', ""))
    y = float(image_name[2].replace('Y', ""))
    return [x, y]

# Find min and max values from test and train labels
def getMinMaxValues(dataDir):
    dataset_examples = os.listdir(dataDir)
    dataset_coords = []

    for data in dataset_examples:
        dataset_coords.append(getLabel(data))

    dataset_coords = np.asarray(dataset_coords)

    min = np.min(dataset_coords, axis=0)
    max = np.max(dataset_coords, axis=0)

    return min[0], min[1], max[0], max[1]

# Get min/max values
minX, minY, maxX, maxY = getMinMaxValues(datasetDir)


# Normalize a coordinate
def normalizeCoords(coords):
    x = (coords[0] - minX) / (maxX - minX)
    y = (coords[1] - minY) / (maxY - minY)
    return [x, y]


# Loads a batch of dataset specifying indices
def getBatch(dataDir, data_list, startIndex, endIndex):
    img_batch = []
    label_batch = []

    index = 0
    for image in data_list:
        if startIndex <= index < endIndex:
            img_batch.append(np.asarray(Image.open(dataDir+image)))
            label_batch.append(normalizeCoords(getAltLabel(image)))
        index += 1

    img_batch = np.asarray(img_batch)
    label_batch = np.asarray(label_batch)

    return img_batch, label_batch

# Model
model = Sequential()
model.add(Conv2D(input_shape=(300, 300, 3), filters=24, kernel_size=(4, 4), strides=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

# Summary
model.summary()

# Compile model
opt = optimizers.RMSprop(lr=0.0005, decay=0.000025)
#opt = optimizers.RMSprop(lr=0.001)
model.load_weights('models/Best/best-net-epoch_2-acc_39.34.h5')
model.compile(loss='mae', optimizer=opt)


# Training
def train_model():
    best_accuracy = 39.34
    for epoch in range(0, epochs):
        print("Epoch ---> " + str(epoch + 1) + "/" + str(epochs))
        batch_time, start_time, end_time = 0, 0, 0

        train_list = shuffleList(__train__)
        test_list = shuffleList(__test__)
        startIndex = 0
        endIndex = batchSize

        num_batches = int(train_examples_num/batchSize)
        for batch in range(0, num_batches):
            start_time = datetime.now()

            img_batch, label_batch = getBatch(__train__, train_list, startIndex, endIndex)
            batch_loss = model.train_on_batch(img_batch, label_batch)
            startIndex = endIndex
            endIndex = startIndex + batchSize
            if not batch_time == 0:
                print("Epoch: " + str(epoch + 1) + "/" + str(epochs)
                      + "    Batch " + str(batch + 1) + "/" + str(num_batches)
                      + "    Batch loss: " + str(batch_loss)
                      + "    Best accuracy: " + str(best_accuracy)
                      + "    Remaining time: "
                      + str(int(batch_time.microseconds/1000000 * (num_batches - batch) / 60))
                      + "m "
                      + str(int((batch_time.microseconds/1000000 * (num_batches - batch)) % 60))
                      + "s")
            else:
                print("Epoch: " + str(epoch + 1) + "/" + str(epochs)
                      + "    Batch " + str(batch + 1) + "/" + str(num_batches)
                      + "    Batch loss: " + str(batch_loss)
                      + "    Best accuracy: " + str(best_accuracy))

            end_time = datetime.now()
            batch_time = end_time - start_time

        print("Calculating predictions on test set...")
        test_predictions, test_labels = calculatePredictions(__test__, test_list, test_examples_num)

        print("Calculating accuracy on test set...\n")

        # Threshold 500m
        # x_thresh = 0.0968141592920358
        # y_thresh = 0.05829173599556346
        test_accuracy = custom_accuracy(test_predictions, test_labels, 0.0968141592920358, 0.05829173599556346)
        print("Test_accuracy on 500m range: " + str(test_accuracy) + "%\n")

        # Threshold 200m
        # x_thresh = 0.03872566371681432
        # y_thresh = 0.023316694398225384
        test_accuracy_200 = custom_accuracy(test_predictions, test_labels, 0.03872566371681432, 0.023316694398225384)
        print("Test_accuracy on 200m range: " + str(test_accuracy_200) + "%\n")

        # Threshold 100m
        # x_thresh = 0.01936283185840716
        # y_thresh = 0.011658347199112692
        test_accuracy_100 = custom_accuracy(test_predictions, test_labels, 0.01936283185840716, 0.011658347199112692)
        print("Test_accuracy on 100m range: " + str(test_accuracy_100) + "%\n")

        if test_accuracy > best_accuracy:
            model.save_weights('models/best-net-epoch_' + str(epoch+1) + '-acc_' + str(test_accuracy) + '.h5',
                               overwrite=True)
            best_accuracy = test_accuracy
            print('Best model saved with accuracy: ' + str(best_accuracy) + '%')
        else:
            print('Accuracy didn\'t improve: ' + str(test_accuracy) + '% is worse than ' + str(best_accuracy) + '%\n')


# Calculate predictions on a set of samples (train/test)
def calculatePredictions(dir, list, num_samples):
    predictions = np.zeros(shape=(1, 2))
    real_labels = np.zeros(shape=(1, 2))

    startIndex = 0
    endIndex = batchSize

    num_batches = int(num_samples / batchSize)
    for batch in range(0, num_batches):
        img_batch, label_batch = getBatch(dir, list, startIndex, endIndex)
        new_pred = model.predict(img_batch, batchSize)
        predictions = np.append(predictions, new_pred, axis=0)
        real_labels = np.append(real_labels, label_batch, axis=0)

        startIndex = endIndex
        endIndex = startIndex + batchSize

    predictions = predictions[1:, :]
    real_labels = real_labels[1:, :]

    return predictions, real_labels


# Calculate accuracy based on a threshold in both latitude and longitude
def custom_accuracy(predictions, real_labels, x_thresh, y_thresh):
    num_correct_predictions = 0
    num_samples, y = predictions.shape

    for i in range(0, num_samples):
        # print("Pred: " + str(predictions[i]) + "    " + "Truth: " + str(real_labels[i]))
        if abs(predictions[i][0] - real_labels[i][0]) < x_thresh and \
           abs(predictions[i][1] - real_labels[i][1]) < y_thresh:
            num_correct_predictions += 1

    accuracy = round(((num_correct_predictions/num_samples)*100), 2)
    return accuracy

# data_augmentation_0_180(trainsetDir, augmented_trainsetDir, train_examples_num)
# data_augmentation_0_180(testsetDir, augmented_testsetDir, test_examples_num)
train_model()

