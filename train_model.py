
from __future__ import print_function
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
import numpy as np
import pickle
from PIL import Image
import os


def getint(name):
    basename = name.partition('.')
    parts = name.partition('_')
    return int(parts[0])

def orderedList(folderPath):
    files = os.listdir(folderPath)
    files.sort(key=getint)
    return files
'''
def normalizeValues(x, y, minX, minY, maxX, maxY):
    values = np.zeros(30000)

    x = (x - minX) / (maxX - minX)
    y = (y - minY) / (maxY - minY)

    return x, y

def deNormalizeValues(x, y, minX, minY, maxX, maxY):
    values = np.zeros(30000)

    x = x * (maxX - minX) + minX
    y = y * (maxY - minY) + minY

    return x, y

def serializeNorm(xy, str, minX, minY, maxX, maxY):
    labels = []
    # for file in os.listdir(file):
    #     file = file.replace(".jpg", "")
    #     file = file.split('_')
    #     x = float(file[1].replace('X', ""))
    #     y = float(file[2].replace('Y', ""))
    for x, y in xy:
        labels.append(normalizeValues(x, y, minX, minY, maxX, maxY))

    with open(str + '.lbl', 'wb') as f:
        pickle.dump(labels, f)

def findMinMaxCoord(trainlabels, testlabels):
    minX = 999.0
    minY = 999.0
    maxX = 0.0
    maxY = 0.0

    for x, y in trainlabels:
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

    for x, y in testlabels:
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

    print(str(minX) + ', ' + str(maxX) + ', ' + str(minY) + ', ' + str(maxY))
    return minX, minY, maxX, maxY

with open('test_labels.lbl', 'rb') as fa:
    trainlabels = pickle.load(fa)
with open('test_labels.lbl', 'rb') as fe:
    testlabels = pickle.load(fe)

minX, minY, maxX, maxY = findMinMaxCoord(trainlabels, testlabels)

# serializeNorm(trainlabels, 'trainNorm', minX, minY, maxX, maxY)

# serializeNorm(testlabels, 'testNorm', minX, minY, maxX, maxY)

with open('trainNorm.lbl', 'rb') as f:
    f = pickle.load(f)
    for i in range(len(f)):
        print(str(f[i]))
with open('testNorm.lbl', 'rb') as f:
    f = pickle.load(f)
    for i in range(len(f)):
        print(str(f[i]))
'''

'''
def extractImages(folder, dim, name):
    name_list = orderedList(folder)
    img_structure = np.empty(dim)

    i = 0
    for file in name_list:
        img = np.asarray(Image.open(folder+file), dtype="uint8")
        np.append(img_structure, img)
        i += 1

    with open(name + '.imgs', 'wb') as f:
        pickle.dump(img_structure, f)

    return img_structure

train_images = extractImages(trainsetDir, 27715, 'trainset')
test_images = extractImages(testsetDir, 6928, 'testset')
'''

batchSize = 64

epochs = 20

num_classes = 2

model_name = 'Bologna_Zones_Model'

dataset = 'bologna_dataset_sparse'

trainsetDir = 'bologna_train_sparse/'

testsetDir = 'bologna_test_sparse/'

with open('trainNorm.lbl', 'rb') as f:
    train_labels = pickle.load(f)

train_labels = np.asarray(train_labels)

with open('testNorm.lbl', 'rb') as f2:
    test_labels = pickle.load(f2)

test_labels = np.asarray(test_labels)

with open('trainset.imgs', 'rb') as f3:
    train_images = pickle.load(f3)

with open('testset.imgs', 'rb') as f4:
    test_images = pickle.load(f4)


# Data generators

#train_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_labels, batch_size=batchSize)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=batchSize)


'''
#base model
base_model = DenseNet121(input_shape=(300, 300, 3), weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(base_model.input, predictions)
print(model.summary())

for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
model.fit_generator(
        train_generator,

        epochs=epochs,

        validation_data=test_generator,

        validation_steps=6936//batchSize,

        steps_per_epoch=27743//batchSize
)
'''


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
input_shape = Input(shape=(160,150,3))

C_1A = Conv2D(filters=64, kernel_size=(2,15), strides=(2,15), activation="elu", kernel_initializer='glorot_normal', name='conv2d_1A')(input_shape)
C_2A = Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),activation="elu", kernel_initializer='glorot_normal', name='conv2d_2A')(C_1A)
P_1A = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2d_1A')(C_2A)
C_3A = Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='glorot_normal', name='conv2d_3A')(P_1A)
FlatA = Flatten()(C_3A)
DropA = Dropout(0.5)(FlatA)
D1A = Dense(128, activation='elu', kernel_initializer='glorot_normal')(DropA)

C_1B = Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), activation="elu", kernel_initializer='glorot_normal', name='conv2d_1B')(input_shape)
C_2B = Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),activation="elu", kernel_initializer='glorot_normal', name='conv2d_2B')(C_1B)
P_1B = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2d_1B')(C_2B)
C_3B = Conv2D(filters=128, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='glorot_normal', name='conv2d_3B')(P_1B)
FlatB = Flatten()(C_3B)
D1B = Dense(128, activation='elu', kernel_initializer='glorot_normal')(FlatB)

Merged = concatenate([D1A, D1B])
Final = Dense(num_classes, activation='softmax')(Merged)

model = Model(input=input_shape, output=Final)
'''

#Summary
model.summary()


#Compile model
optimizer = 'rmsprop'
model.compile(loss='mean_squared_error',

              optimizer=optimizer,

              metrics=['accuracy'])



#Training
model.fit_generator(

        train_generator,

        epochs=epochs,

        validation_data=test_generator,

        validation_steps=6928//batchSize,

        steps_per_epoch=27715//batchSize)

'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
