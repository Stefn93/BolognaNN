from __future__ import print_function

import pickle

import keras
import tensorflow as tf
from PIL import Image
from keras.layers import *
from keras.models import *
import keras.callbacks

model_name = 'Bologna_Zones_Model'
dataset = 'bologna_dataset_sparse'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'

batchSize = 32
epochs = 40
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
    batchSize_tensor = tf.fill([1, 1], batchSize)
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
model.add(Conv2D(input_shape=(300, 300, 3), filters=32, kernel_size=(4, 4), strides=(2, 2)))
model.add(Conv2D(filters=24, kernel_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.5))
# model.add(Activation('sigmoid'))
model.add(Dense(2))

#Summary
model.summary()

#Compile model
opt = optimizers.RMSprop(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=[custom_accuracy])
#model.compile(loss="mean_squared_error", optimizer='rmsprop')


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

    return str((acc/tot_acc)*100)


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
history.init_names(testsetDir, image_name_list_test, test_labels)

save = keras.callbacks.ModelCheckpoint('models/best_net-{val_custom_accuracy:.2f}.hdf5', monitor='val_custom_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)

model.fit_generator(dataGenerator(trainsetDir, image_name_list_train, train_labels, batchSize),
                    validation_data=dataGenerator(testsetDir, image_name_list_test, test_labels, batchSize),
                    steps_per_epoch=train_examples//batchSize,
                    validation_steps=test_examples//batchSize,
                    epochs=epochs,
                    callbacks=[history, save])

print(history.train_res)
print(history.test_res)

'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
