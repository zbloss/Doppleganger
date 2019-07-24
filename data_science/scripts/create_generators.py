import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator




# Number of classes
train_list = open('C:/data/vgg_face2/train_list.txt', 'rb').read().decode()
test_list = open('C:/data/vgg_face2/test_list.txt', 'rb').read().decode()

sample = train_list[0:1000]

dirs = os.walk('C:/data/vgg_face2/')

test_dirs = next(os.walk('C:/data/vgg_face2/test'))[1]
train_dirs = next(os.walk('C:/data/vgg_face2/train'))[1]

all_dirs = test_dirs + train_dirs
all_dirs = list(set(all_dirs))


num_classes = len(all_dirs)
print(f'num_classes: {num_classes}')



print('Creating Generators...')
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:/data/vgg_face2/train/',
        target_size=(175, 175),
        batch_size=128, 
        class_mode='categorical',
        color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
        'C:/data/vgg_face2/test/',
        target_size=(175, 175),
        batch_size=128,
        class_mode='categorical',
        color_mode='grayscale')

pickle.dump(train_generator, open('../data/generators/train_gen.pkl', 'wb'))
pickle.dump(validation_generator, open('../data/generators/val_gen.pkl', 'wb'))