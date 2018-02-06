
from __future__ import print_function
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *



import numpy as np
import pickle

import os


batchSize = 64

epochs = 5

num_classes = 6

model_name = 'Bologna_Zones_Model'

dataset = 'bologna_dataset_sparse'

trainsetDir = 'bologna_train/'

testsetDir = 'bologna_test/'

with open('train_labels.lbl', 'rb') as f:
    train_labels = pickle.load(f)

train_labels = np.asarray(train_labels)

with open('test_labels.lbl', 'rb') as f:
    test_labels = pickle.load(f)

test_labels = np.asarray(test_labels)


# Data generators

#train_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow()

#train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(300,300), shuffle=True)
#test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(300,300))

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

'''
model = Sequential()

model.add(Conv2D(input_shape=(300, 300, 3), filters=16, kernel_size=(5,5), strides=(5,5), activation="elu", kernel_initializer='glorot_uniform', dilation_rate=1))
model.add(Conv2D(filters=24, kernel_size=(3,3), strides=(3,3),activation="elu", kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=32, kernel_size=(2,2), strides=(2,2),activation="elu", kernel_initializer='glorot_uniform'))
model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1,1),activation="elu", kernel_initializer='glorot_uniform'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform'))
model.add(Dense(num_classes, activation='softmax'))
'''

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

#Compile model

#optimizer = 'adadelta'
'''
optimizer = 'rmsprop'
model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



#Summary

model.summary()



#Training
#model.load_weights('arch_weights/MuGeRe Weights_F5.h5', by_name=True)
model.fit_generator(

        train_generator,

        epochs=epochs,

        validation_data=test_generator,

        validation_steps=6940//batchSize,

        steps_per_epoch=27739//batchSize)


# serialize model to YAML
model_yaml = model.to_yaml()
with open("Bologna_Architecture_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("Bologna_Weights_1.h5")
print("Saved model to disk")
'''
