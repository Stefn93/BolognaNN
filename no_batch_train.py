from __future__ import print_function

import pickle

import keras
import tensorflow as tf
from PIL import Image
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint

model_name = 'Bologna_Zones_Model'
dataset = 'bologna_dataset_sparse'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'

epochs = 10
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

def loadLabels():
    with open('trainNorm.lbl', 'rb') as f:
        train_labels = pickle.load(f)

    train_labels = np.asarray(train_labels)

    with open('testNorm.lbl', 'rb') as f:
        test_labels = pickle.load(f)

    test_labels = np.asarray(test_labels)

    return train_labels, test_labels

def extract_label(labels, index):
    i = 0
    for label in labels:
        if i == index:
            return label
        i += 1


def extract_image(data_folder, data_name):
    img = np.asarray(Image.open(data_folder+data_name))
    return img


def data_generator(dir, img_names, labels):
    length = len(img_names)

    # this line is just to make the generator infinite, keras needs that
    while True:
        index = 0
        while index < length:
            X = extract_image(dir, img_names[index])
            Y = extract_label(labels, index)
            yield (X, Y)

def calcAcc(dir, img_names, labels):
    l = len(img_names)
    index = 0
    acc = 0
    tot_acc = 0
    x_thresh = 0.0968141592920358
    y_thresh = 0.05829173599556346

    while index < l:
        X = extract_image(dir, img_names[index])
        Y = extract_label(labels, index)

        pred = model.predict(X)
        if abs(pred[index][0] - Y[index][0]) < x_thresh and abs(pred[index][1] - Y[index][1]) < y_thresh:
            acc += 1
        tot_acc += 1

        print('predizione: ' + str(pred[index]) + ' , truth: ' + str(Y[index]))
        print('Accuracy: ' + str((acc / tot_acc) * 100) + '%\n')
        index += 1

    return 'Accuracy: ' + str((acc / tot_acc) * 100) + '%'

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


# Model
model = Sequential()
model.add(Conv2D(input_shape=(300, 300, 3), filters=32, kernel_size=(5, 5), strides=(5, 5), activation='sigmoid'))
model.add(Conv2D(filters=24, kernel_size=(3, 3), strides=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.5))
# model.add(Activation('sigmoid'))
model.add(Dense(2))

# Summary
model.summary()


# Compile model
# model.compile(loss=custom_loss, optimizer='rmsprop', metrics=[custom_accuracy])
model.compile(loss="mean_squared_error", optimizer='rmsprop')


# Training
image_name_list_train = orderedList(trainsetDir)
image_name_list_test = orderedList(testsetDir)
train_labels, test_labels = loadLabels()

train_history = Histories()
test_history = Histories()
train_history.init_names(trainsetDir, image_name_list_train, train_labels)
test_history.init_names(testsetDir, image_name_list_test, test_labels)


model.fit_generator(data_generator(trainsetDir, image_name_list_train, train_labels),
                    validation_data=data_generator(testsetDir, image_name_list_test, test_labels),
                    steps_per_epoch=train_examples,
                    validation_steps=test_examples,
                    epochs=epochs,
                    callbacks=[train_history, test_history, ModelCheckpoint('model.h5', save_best_only=True)]
                    )

print(train_history.train_res)

