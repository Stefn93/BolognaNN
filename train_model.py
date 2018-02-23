from __future__ import print_function

import os
import pickle

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import *
from keras.models import *

model_name = 'Bologna_Zones_Model'
dataset = 'bologna_dataset_sparse'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'

batchSize = 32
epochs = 5
num_classes = 2
train_examples = 27715
test_examples = 6928


def getint(name):
    basename = name.partition('.')
    parts = name.partition('_')
    return int(parts[0])


def orderedList(folderPath):
    files = os.listdir(folderPath)
    files.sort(key=getint)
    return files


def extractBatch(dataFolder, data_names):
    img_batch = []

    for image in data_names:
            img = np.asarray(Image.open(dataFolder+image))
            img_batch.append(img)

    img_batch = np.array(img_batch)
    return img_batch


def extractLabelBatch(labels, startIndex, end_index):
    label_batch = []

    i = 0
    k = 0
    for label in labels:
        if(i >= startIndex and i < end_index):
            label_batch.append(label)
            k += 1
        i+=1
        if(k==batchSize): break;

    label_batch = np.array(label_batch)
    return label_batch


def loadLabels():
    with open('trainNorm.lbl', 'rb') as f:
        train_labels = pickle.load(f)

    train_labels = np.asarray(train_labels)

    with open('testNorm.lbl', 'rb') as f:
        test_labels = pickle.load(f)

    test_labels = np.asarray(test_labels)

    return train_labels, test_labels


# Custom loss function, euclidean distance between normalized coordinates
def custom_loss(y_true, y_pred):
    x_diff = y_true[:][0] - y_pred[:][0]
    y_diff = y_true[:][1] - y_pred[:][1]
    x_diff_square = K.square(x_diff)
    y_diff_square = K.square(y_diff)
    xy_sum = x_diff_square + y_diff_square
    return K.sum(K.sqrt(xy_sum))


# Custom accuracy
def custom_accuracy(y_true, y_pred):
    batchSize_tensor = tf.fill([1,1], batchSize)
    x_thresh = tf.fill([batchSize, 1], 0.0968141592920358)
    y_thresh = tf.fill([batchSize, 1], 0.05829173599556346)

    x_diff = tf.sqrt(tf.square(tf.subtract(y_true[:][0], y_pred[:][0])))
    y_diff = tf.sqrt(tf.square(tf.subtract(y_true[:][1], y_pred[:][1])))

    x_bool = tf.less(x_diff, x_thresh)
    y_bool = tf.less(y_diff, y_thresh)

    xy_bool = tf.logical_and(x_bool, y_bool)
    n_valid_points = tf.reduce_sum(tf.to_int32(xy_bool))
    res = tf.divide(n_valid_points, batchSize_tensor)
    return res


def dataGenerator(dir, img_names, labels, batch_size):
    L = len(img_names)

    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_end < L:
            limit = min(batch_end, L)
            X = extractBatch(dir, img_names[batch_start:limit])
            Y = extractLabelBatch(labels, batch_start, limit)

            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


# Model
model = Sequential()
model.add(Conv2D(input_shape=(300, 300, 3), filters=16, kernel_size=(5, 5), strides=(3, 3), activation="elu",
                 kernel_initializer='he_normal'))
model.add(Conv2D(filters=24, kernel_size=(3, 3), strides=(3, 3), activation="elu", kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="elu", kernel_initializer='he_normal'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="elu", kernel_initializer='he_normal'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='elu', kernel_initializer='he_normal'))
# model.add(Activation('sigmoid'))
model.add(Dense(2))

'''
input_shape = Input(shape=(160,150,3))

def high_features(input_shape):
    conv1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(5, 5), activation="elu",
                  kernel_initializer='he_normal')(input_shape)
    conv2 = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="elu",
                   kernel_initializer='he_normal')(conv1)
    conv_flat = Flatten()(conv2)
    return conv_flat

def mid_features(input_shape):
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(3, 3), activation="elu",
                  kernel_initializer='he_normal')(input_shape)
    conv2 = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="elu",
                   kernel_initializer='he_normal')(conv1)
    conv_flat = Flatten()(conv2)
    return conv_flat

def low_features(input_shape):
    conv1 = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="elu",
                  kernel_initializer='he_normal')(input_shape)
    conv2 = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="elu",
                   kernel_initializer='he_normal')(conv1)
    conv_flat = Flatten()(conv2)
    return conv_flat

conv1 = high_features(input_shape)
conv2 = mid_features(input_shape)
conv3 = low_features(input_shape)
merged = concatenate([conv1, conv2, conv3])
pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(merged)
conv4 = Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='he_normal')(pool1)
flat1 = Flatten()(conv4)
drop1 = Dropout(0.5)(flat1)
dense1 = Dense(128, activation='elu', kernel_initializer='he_normal')(drop1)
final = Dense(num_classes, activation='sigmoid')(dense1)

model = Model(input=input_shape, output=final)
'''

#Summary
model.summary()


#Compile model
# model.compile(loss=custom_loss, optimizer='rmsprop', metrics=[custom_accuracy])
model.compile(loss="mean_squared_error", optimizer='rmsprop')


def calcAcc(dir, img_names, labels):
    l = len(img_names)
    batch_start = 0
    batch_end = batchSize
    acc = 0
    tot_acc = 0

    while batch_end < l:
        limit = min(batch_end, l)
        x_thresh = 0.0968141592920358
        y_thresh = 0.05829173599556346

        X = extractBatch(dir, img_names[batch_start:limit])
        Y = extractLabelBatch(labels, batch_start, limit)

        pred = model.predict(X, batchSize)
        # print('Pred: ' + str(pred) + ', shape: ' + str(pred.shape))
        for i in range(batchSize):
            if abs(pred[i][0] - Y[i][0]) < x_thresh and abs(pred[i][1] - Y[i][1]) < y_thresh:
                acc += 1
            tot_acc += 1
            print('predizione: ' + str(pred[i]) + ' , truth: ' + str(Y[i]))

        print('Accuracy: ' + str((acc/tot_acc)*100) + '%\n')
        batch_start += batchSize
        batch_end += batchSize

    return 'Accuracy: ' + str((acc/tot_acc)*100) + '%\n'

class Histories(keras.callbacks.Callback):
    dir = ''
    img_names = ()
    labels = ()
    train_res = []
    test_res = []

    def init_names(self, dir, img_names, labels):
        self.dir = dir
        self.img_names = img_names
        self.labels = labels

    def on_epoch_end(self, epochs, logs={}):
        if self.dir == trainsetDir:
            self.train_res.append(calcAcc(self.dir, self.img_names, self.labels))
        elif self.dir == testsetDir:
            self.test_res.append(calcAcc(self.dir, self.img_names, self.labels))
        return


# Training
image_name_list_train = orderedList(trainsetDir)
image_name_list_test = orderedList(testsetDir)
train_labels, test_labels = loadLabels()

history = Histories()
history.init_names(trainsetDir, image_name_list_train, train_labels)
#Histories.init_names(Histories, testsetDir, image_name_list_test, test_labels)


model.fit_generator(dataGenerator(trainsetDir, image_name_list_train, train_labels, batchSize),
                    validation_data=dataGenerator(testsetDir, image_name_list_test, test_labels, batchSize),
                    steps_per_epoch=train_examples//batchSize,
                    validation_steps=test_examples//batchSize,
                    epochs=epochs,
                    callbacks=[history])

print(history.train_res)


'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
