# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:15:18 2016

@author: Ulkesh
"""


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import os


from keras import backend as K
K.set_image_dim_ordering('th')


# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'train'
validation_data_dir = 'val'
nb_classes = 5
nb_train_samples = 40000
nb_validation_samples = 2000
nb_epoch = 20


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
print(model.summary())

# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(rescale=1./255, zca_whitening=True, rotation_range=90.0, 
#                                   horizontal_flip=True, vertical_flip=True)

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True,
                                    width_shift_range=0.2, height_shift_range=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True,
                                  width_shift_range=0.2, height_shift_range=0.2)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size=50,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size=50,
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose = 2,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
        

i=0
while 1:
    i = i+1
    if os.path.isfile(str(i) + '_try.h5') == False:
        model.save_weights(str(i) + '_try.h5')
        print (i, "try")
        break;
        



#import cv2
#
#im = cv2.imread('train\Gesture_1\1_74.jpg')
#cv2.imshow("resized", im)
#cv2.waitKey(0)
#print(im.shape)