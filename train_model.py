
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

batchSize = 64

epochs = 20

num_classes = 2

model_name = 'Bologna_Zones_Model'

dataset = 'bologna_dataset_sparse'

trainsetDir = 'bologna_train_sparse/'

testsetDir = 'bologna_test_sparse/'


def getint(name):
    basename = name.partition('.')
    parts = name.partition('_')
    return int(parts[0])

def orderedList(folderPath):
    files = os.listdir(folderPath)
    files.sort(key=getint)
    return files

def extractImages(folder, name):
    name_list = orderedList(folder)
    img_structure = []

    i = 0
    for file in name_list:
        img = np.asarray(Image.open(folder+file))
        img_structure.append(img)
        i+=1
        print('Img ' + str(i))

    #img_structure = np.array(img_structure)

    with open(name + '.imgs', 'wb') as f:
        pickle.dump(img_structure, f)

    return img_structure

train_images = np.array(extractImages(trainsetDir, 'trainset'))
test_images = np.array(extractImages(testsetDir, 'testset'))



def loadDataset():
    with open('trainNorm.lbl', 'rb') as f:
        train_labels = pickle.load(f)

    train_labels = np.asarray(train_labels)

    with open('testNorm.lbl', 'rb') as f:
        test_labels = pickle.load(f)

    test_labels = np.asarray(test_labels)

    with open('trainset.imgs', 'rb') as f:
        train_images = pickle.load(f)

    with open('testset.imgs', 'rb') as f:
        test_images = pickle.load(f)

    return train_images, train_labels, test_images, test_labels

# Data generators

#train_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow(train_images, train_labels, batch_size=batchSize)
#test_generator = test_datagen.flow(test_images, test_labels, batch_size=batchSize)




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
#model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])



#Training
#model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator, validation_steps=6928//batchSize, steps_per_epoch=27715//batchSize)


'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
