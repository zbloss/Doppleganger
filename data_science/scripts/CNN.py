import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime


import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense

print(f'GPU: {tf.test.is_gpu_available()}')


# Number of classes
train_list = open('D:/data/vgg_face2/train_list.txt', 'rb').read().decode()
test_list = open('D:/data/vgg_face2/test_list.txt', 'rb').read().decode()

sample = train_list[0:1000]

dirs = os.walk('D:/data/vgg_face2/')

test_dirs = next(os.walk('D:/data/vgg_face2/test'))[1]
train_dirs = next(os.walk('D:/data/vgg_face2/train'))[1]

all_dirs = test_dirs + train_dirs
all_dirs = list(set(all_dirs))


num_classes = len(all_dirs)
print(f'num_classes: {num_classes}')

model = Sequential()

model.add(Conv2D(32, kernel_size=5,input_shape=(224, 224, 1), activation = 'relu'))
model.add(Conv2D(32, kernel_size=5, activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=3, activation = 'relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))

model.add(Dense(8631, activation = "softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print('Creating Generators...')
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'D:/data/vgg_face2/train/',
        target_size=(224, 224),
        batch_size=256, 
        class_mode='categorical',
        color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
        'D:/data/vgg_face2/test/',
        target_size=(224, 224),
        batch_size=256,
        class_mode='categorical',
        color_mode='grayscale')

# Callbacks
tb = TensorBoard(log_dir='../logs')
checks = ModelCheckpoint(filepath='../models/best_model.h5', save_best_only=True, monitor='val_acc', mode='max')

print('Training...')
start = datetime.now()
model.fit_generator(
        train_generator,
        epochs=500,
        validation_data=validation_generator,
        callbacks=[tb, checks])

print(f'That took: {datetime.now() - start}')