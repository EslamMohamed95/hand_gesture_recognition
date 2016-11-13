# Import Statements
import numpy as numpy
import pandas as pd
from random import shuffle
from keras.models import Model
from keras.layers import Dense, Input, Convolution2D, Dropout, MaxPooling2D, Merge, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import cv2, os


# GLOBAL VARIABLES
PATH_BASE = 'data'
EXT_TRAIN = 'train'
EXT_VAL = 'val'
EXT_TEST = 'test'
BATCH_SIZE = 10


# Layers Convolution size
CONV_1_SIZE = 5
CONV_2_SIZE = 5
CONV_3_SIZE = 5
CONV_4_SIZE = 5
CONV_5_SIZE = 3
CONV_6_SIZE = 3


# Model
def create_model(input_shape):
    
    inp = Input(shape=input_shape, name='Input')
    
    conv_1 = Convolution2D(32, CONV_1_SIZE, CONV_1_SIZE, init='glorot_uniform', border_mode='same', name='conv_1_{}x{}'.format(CONV_1_SIZE, CONV_1_SIZE))(inp)
    batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(conv_1)
    relu_1 = Activation('relu')(batch_normalization_1)
    
    conv_2 = Convolution2D(32, CONV_2_SIZE, CONV_2_SIZE, init='glorot_uniform', border_mode='same', name='conv_2_{}x{}'.format(CONV_2_SIZE, CONV_2_SIZE))(relu_1)
    batch_normalization_2 = BatchNormalization(name='batch_normalization_2')(conv_2)
    relu_2 = Activation('relu')(batch_normalization_2)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(relu_2)
    dropout_1 = Dropout(0.2)(max_pool_1)
    
    
    conv_3 = Convolution2D(64, CONV_3_SIZE, CONV_3_SIZE, init='glorot_uniform', border_mode='same', name='conv_3_{}x{}'.format(CONV_3_SIZE, CONV_3_SIZE))(dropout_1)
    batch_normalization_3 = BatchNormalization(name='batch_normalization_3')(conv_3)
    relu_3 = Activation('relu')(batch_normalization_3)
    
    conv_4 = Convolution2D(64, CONV_4_SIZE, CONV_4_SIZE, init='glorot_uniform', border_mode='same', name='conv_4_{}x{}'.format(CONV_4_SIZE, CONV_4_SIZE))(relu_3)
    batch_normalization_4 = BatchNormalization(name='batch_normalization_4')(conv_4)
    relu_4 = Activation('relu')(batch_normalization_4)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(relu_4)
    dropout_2 = Dropout(0.2)(max_pool_2)
    
    
    conv_5 = Convolution2D(32, CONV_5_SIZE, CONV_5_SIZE, init='glorot_uniform', border_mode='same', name='conv_5_{}x{}'.format(CONV_5_SIZE, CONV_5_SIZE))(dropout_2)
    batch_normalization_5 = BatchNormalization(name='batch_normalization_5')(conv_5)
    relu_5 = Activation('relu')(batch_normalization_5)
    
    conv_6 = Convolution2D(32, CONV_6_SIZE, CONV_6_SIZE, init='glorot_uniform', border_mode='same', name='conv_6_{}x{}'.format(CONV_6_SIZE, CONV_6_SIZE))(relu_5)
    batch_normalization_6 = BatchNormalization(name='batch_normalization_6')(conv_6)
    relu_6 = Activation('relu')(batch_normalization_6)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2), name='max_pool_3')(relu_6)
    dropout_3 = Dropout(0.2)(max_pool_3)


    flatten = Flatten()(dropout_3)
    dense_1 = Dense(256, name='dense_1')(flatten)
    batch_normalization_7 = BatchNormalization(name='batch_normalization_7')(dense_1)
    relu_7 = Activation('relu')(batch_normalization_7)
    dropout_4 = Dropout(0.2)(relu_7)
    
    dense_2 = Dense(128, name='dense_2')(relu_7)
    batch_normalization_8 = BatchNormalization(name='batch_normalization_8')(dense_2)
    relu_8 = Activation('relu')(batch_normalization_8)
    dropout_5 = Dropout(0.2)(relu_8)
    
    out = Dense(10, activation='softmax', name='Output')(dropout_5)

    model = Model(input = inp, output = out)

    return model

model = create_model((1, 48, 48))
#print model.summary()


def batch_generator(train_files, val_files, batch_size):
    for i in range(0, len(train_files), batch_size):
        yield train_files[i:i+batch_size], [int(x[0]) for x in train_files[i:i+batch_size]], val_files[i/2:i/2+batch_size], [int(x[0]) for x in val_files[i/2:i/2+batch_size]]


# Getting Data, making batches and training on thoe batches
train_files = os.listdir(os.path.join(PATH_BASE, EXT_TRAIN))
val_files = os.listdir(os.path.join(PATH_BASE, EXT_VAL))
test_files = os.listdir(os.path.join(PATH_BASE, EXT_TEST))

shuffle(train_files)
shuffle(val_files)
shuffle(test_files)
shuffle(train_files)
shuffle(val_files)
shuffle(test_files)

print '\nNumber of training examples: {}'.format(len(train_files))
print 'Number of validation examples: {}'.format(len(val_files))
print 'Number of testing examples: {}\n'.format(len(test_files))

for batch_X_train, batch_Y_train, batch_X_val, batch_Y_val in batch_generator(train_files, val_files, BATCH_SIZE):
    print len(batch_X_train), len(batch_Y_train), len(batch_X_val), len(batch_Y_val)


