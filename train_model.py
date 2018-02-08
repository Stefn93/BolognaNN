
from __future__ import print_function
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
import keras.backend as K
import numpy as np
import pickle
from PIL import Image
import os

model_name = 'Bologna_Zones_Model'
dataset = 'bologna_dataset_sparse'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'

batchSize = 32
epochs = 20
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

def extractBatch(dataFolder, data_names, startIndex):
    img_batch = []

    i = 0
    k = 0
    for image in data_names:
        if(i >= startIndex and i < startIndex + batchSize):
            img = np.asarray(Image.open(dataFolder+image))
            img_batch.append(img)
            k += 1
        i+=1
        if(k==batchSize): break;

    img_batch = np.array(img_batch)
    return img_batch

def extractLabelBatch(labels, startIndex):
    label_batch = []

    i = 0
    k = 0
    for label in labels:
        if(i >= startIndex and i < startIndex + batchSize):
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

def generateTrainingExamples(trainDir):
    image_name_list = orderedList(trainDir)
    train_labels, test_labels = loadLabels()

    while 1:
        for img_name, img_label in zip(image_name_list, train_labels):
            img = np.asarray(Image.open(trainDir + img_name))
            yield (img, img_label)

def generateTestExamples(testDir):
    image_name_list = orderedList(testDir)
    train_labels, test_labels = loadLabels()

    while 1:
        for img_name, img_label in zip(image_name_list, test_labels):
            img = np.asarray(Image.open(testDir + img_name))
            yield (img, img_label)


def startTraining():
    image_name_list = orderedList(trainsetDir)

    for i in range(int(train_examples/batchSize)):
        train_img_batch = extractBatch(trainsetDir, image_name_list, i*batchSize)
        train_labels, test_labels = loadLabels()
        train_label_batch = extractLabelBatch(train_labels, i*batchSize)
        #model.train_on_batch(train_img_batch, train_label_batch)
        model.fit(train_img_batch, train_label_batch)
        #print("Batch " + str(i+1))


#Model
model = Sequential()
model.add(Conv2D(input_shape=(300, 300, 3), filters=16, kernel_size=(5,5), strides=(5,5), activation="elu", kernel_initializer='he_normal'))
model.add(Conv2D(filters=24, kernel_size=(3,3), strides=(3,3),activation="elu", kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=32, kernel_size=(2,2), strides=(2,2),activation="elu", kernel_initializer='he_normal'))
model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),activation="elu", kernel_initializer='he_normal'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='elu', kernel_initializer='he_normal'))
model.add(Dense(2, activation='sigmoid'))

#Summary
model.summary()


#Compile model
def custom_mse(y_true,y_pred):

    y_true_x = []
    y_true_y = []
    y_pred_x = []
    y_pred_y = []
    y_calc_x = np.zeros(batchSize)
    y_calc_y = np.zeros(batchSize)

    print(y_true.shape)
    print(y_pred.shape)
    for i in range(batchSize):
        y_true_x.append(y_true[i][0])
        y_true_y.append(y_true[i][1])

        y_pred_x.append(y_pred[i][0])
        y_pred_y.append(y_pred[i][1])

        y_calc_x[i] = y_true_x[i] - y_pred_x[i]
        y_calc_y[i] = y_true_y[i] - y_pred_y[i]

    res_x = K.mean(y_calc_x)
    res_y = K.mean(y_calc_y)

    return K.sqrt(K.square(res_x) + K.square(res_y))

model.compile(loss=custom_mse, optimizer='rmsprop', metrics=['accuracy'])


startTraining()
#Training
# model.fit_generator(generateTrainingExamples(trainsetDir),
#                     epochs=epochs,
#                     validation_data=generateTestExamples(testsetDir),
#                     steps_per_epoch=train_examples//batchSize,
#                     validation_steps=test_examples//batchSize)


'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
