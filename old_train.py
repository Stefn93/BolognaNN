from __future__ import print_function

import pickle
import keras
import tensorflow as tf
from PIL import Image
from keras.layers import *
from keras.models import *
import keras.callbacks
import re
from random import shuffle

model_name = 'Bologna_Zones_Model'
dataset = 'bologna_dataset_sparse'
trainsetDir = 'bologna_train_sparse/'
testsetDir = 'bologna_test_sparse/'

batchSize = 32
epochs = 40
num_classes = 2
train_examples = 27715
test_examples = 6928
glob_acc = 0.0

def getint(name):
    basename = name.partition('.')
    parts = name.partition('_')
    return int(parts[0])

def getTrainSet(dirName):
    img_list = []
    lbl_list = []
    files = os.listdir(dirName)
    shuffle(files)
    print(len(files))
    for f in files[:256]:
        #print(getLabel(f))
        lbl_list.append(getLabel(f))
        img = np.asarray(Image.open(dirName + f))
        img_list.append(img)
    lbl_array = np.array(lbl_list)
    lbl_min_x= np.min(lbl_array[:, 0])
    lbl_max_x = np.max(lbl_array[:, 0])
    lbl_min_y = np.min(lbl_array[:, 1])
    lbl_max_y = np.max(lbl_array[:, 1])
    lbl_array[:, 0] = (lbl_array[:, 0] - lbl_min_x) / (lbl_max_x - lbl_min_x)
    lbl_array[:, 1] = (lbl_array[:, 1] - lbl_min_y) / (lbl_max_y - lbl_min_y)
    print(lbl_array[0:50])
    return np.array(img_list), np.array(lbl_list)

def orderedList(folderPath):
    files = os.listdir(folderPath)
    files.sort(key=getint)
    return files

def shuffleList(folderPath):
    files = os.listdir(folderPath)
    files = shuffle(files)
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

def getLabel(image_name):
    image_name = image_name.replace(".jpg", "")
    image_name = image_name.split('_')
    x = float(image_name[1].replace('X', ""))
    y = float(image_name[2].replace('Y', ""))
    coord = [x, y]
    return coord

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
model.add(Conv2D(input_shape=(300, 300, 3), filters=24, kernel_size=(4, 4), strides=(4, 4)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))


#Summary
model.summary()

#Compile model
opt = optimizers.Adam(lr=0.0001, decay=0.0001)
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

        print("shape: " + str(Y.shape))
        print("media asse 1: " + str(np.mean(Y[:, 0])))
        print("media asse 2: " + str(np.mean(Y[:, 1])))

        pred = model.predict(X, batchSize)
        # print('Pred: ' + str(pred) + ', shape: ' + str(pred.shape))
        for i in range(batchSize):
            #0eal_coords = getLabel(X[i])
            if abs(pred[i][0] - Y[i][0]) < x_thresh and abs(pred[i][1] - Y[i][1]) < y_thresh:
                acc += 1
            tot_acc += 1
            print('predizione: ' + str(pred[i]) + ' , truth: ' + str(Y[i]))

        glob_acc = round(((acc/tot_acc)*100), 2)
        print('Accuracy: ' + str(glob_acc) + '%\n')
        batch_start += batchSize
        batch_end += batchSize

    # files = os.listdir('models')
    # if (len(files) > 0):
    #     for file in files:
    #         s = re.sub("bestnet", "", file)
    #         s = re.sub("\.h5", "", s)
    #         if(glob_acc > float(s)):
    #             model.save_weights("models/bestnet" + str(glob_acc) + ".h5")
    #             os.remove("models/"+file)
    # else:
    #     model.save_weights("models/bestnet" + str(glob_acc) + ".h5")

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

        #image_name_list_train = shuffleList(trainsetDir)
        return


# Training
#image_name_list_train = orderedList(trainsetDir)
#image_name_list_test = orderedList(testsetDir)
#train_labels, test_labels = loadLabels()

image_name_list_train, train_labels = getTrainSet(trainsetDir)
image_name_list_test, test_labels = getTrainSet(testsetDir)

history = Histories()
history.init_names(trainsetDir, image_name_list_train, train_labels)
history.init_names(testsetDir, image_name_list_test, test_labels)

#save = keras.callbacks.ModelCheckpoint('models/best_net-'+str(glob_acc)+'.hdf5', monitor=glob_acc, verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)

# model.fit_generator(dataGenerator(trainsetDir, image_name_list_train, train_labels, batchSize),
#                     validation_data=dataGenerator(testsetDir, image_name_list_test, test_labels, batchSize),
#                     #steps_per_epoch=train_examples//batchSize,
#                     steps_per_epoch=10,
#                     validation_steps=test_examples//batchSize,
#                     epochs=epochs,
#                     #callbacks=[history, save])
#                     callbacks=[history])
print("hey")
model.fit(  image_name_list_train,
            train_labels,
            validation_data=(image_name_list_test, test_labels),
            #steps_per_epoch=train_examples//batchSize,
            steps_per_epoch=10,
            validation_steps=test_examples//batchSize,
            epochs=epochs,
            #callbacks=[history, save])
            callbacks=[history])

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
