
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
import tensorflow as tf

model_name = 'Bologna_Zones_Model'
dataset = 'bologna_dataset_sparse'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'

batchSize = 32
epochs = 1
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

#Custom loss function, euclidean distance between normalized coordinates
def custom_loss(y_true, y_pred):
    x_diff = y_true[:][0] - y_pred[:][0]
    y_diff = y_true[:][1] - y_pred[:][1]
    x_diff_square = K.square(x_diff)
    y_diff_square = K.square(y_diff)
    xy_sum = x_diff_square + y_diff_square
    return K.mean(K.sqrt(xy_sum))

#Custom accuracy
def custom_accuracy(y_true, y_pred):
    size_tensor = tf.divide(tf.size(y_true), tf.fill([1,1], 2))
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

'''
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
'''

#base model
base_model = DenseNet121(input_shape=(300, 300, 3), weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(2, activation='sigmoid')(x)

model = Model(base_model.input, predictions)

for layer in base_model.layers:
    layer.trainable = False

#Summary
model.summary()


#Compile model
model.compile(loss=custom_loss, optimizer='rmsprop', metrics=[custom_accuracy])


#Training
image_name_list_train = orderedList(trainsetDir)
image_name_list_test = orderedList(testsetDir)
train_labels, test_labels = loadLabels()

def dataGenerator(dir, img_names, labels, batch_size):
    L = len(img_names)

    #this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = extractBatch(dir, img_names[batch_start:limit])
            Y = extractLabelBatch(labels, batch_start, limit)

            yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size


model.fit_generator(dataGenerator(trainsetDir, image_name_list_train, train_labels, batchSize),
                    validation_data=dataGenerator(testsetDir, image_name_list_test, test_labels, batchSize),
                    steps_per_epoch=train_examples//batchSize,
                    validation_steps=test_examples//batchSize,
                    epochs=2)
'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
