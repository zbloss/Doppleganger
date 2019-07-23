import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime


import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

model = NASNetLarge(weights=None, classes=num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
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
        target_size=(331, 331),
        batch_size=4, 
        class_mode='categorical',
        color_mode='rgb')

validation_generator = test_datagen.flow_from_directory(
        'D:/data/vgg_face2/test/',
        target_size=(331, 331),
        batch_size=4,
        class_mode='categorical',
        color_mode='rgb')

tb = TensorBoard(log_dir='../logs')

print('Training...')
start = datetime.now()
model.fit_generator(
        train_generator,
        steps_per_epoch=4,
        epochs=5000,
        validation_data=validation_generator,
        validation_steps=4,
        callbacks=[tb])

print(f'That took: {datetime.now() - start}')