# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:48:06 2016

@author: syamprasadkr
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
import cv2
import numpy as np
import matplotlib.pyplot as plt
import theano
import os
from keras import backend as K
K.set_image_dim_ordering('th')


PATH = 'data'
EXT1 = 'train'
EXT2 = 'val'
EXT3 = 'test'
PATH1 = os.path.join(PATH, EXT1)
PATH2 = os.path.join(PATH, EXT2)
PATH3 = os.path.join(PATH, EXT3)

img_width, img_height = 64, 64

nb_train_samples = 362087
nb_validation_samples = 181043
nb_epoch =3 

#model
model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape = (1, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Convolution2D(256, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Convolution2D(512, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)
                                 
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(PATH1,
                                                    target_size = (img_width, img_height),
                                                    color_mode = 'grayscale',                                                    
                                                    batch_size = 100,
                                                    class_mode = 'categorical')  
                                    
validation_generator = test_datagen.flow_from_directory(PATH2,
                                                    target_size = (img_width, img_height),
                                                    color_mode = 'grayscale',                                                    
                                                    batch_size = 100,
                                                    class_mode = 'categorical')                                    
                

model.fit_generator(train_generator,
                    samples_per_epoch = nb_train_samples,
                    nb_epoch = nb_epoch,
                    validation_data = validation_generator,
                    nb_val_samples = nb_validation_samples)

model.save_weights('3_hgm.h5')

#score = model.evaluate(X_test, y_test, batch_size = 300)


 




