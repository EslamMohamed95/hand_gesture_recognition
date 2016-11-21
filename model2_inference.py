# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:01:32 2016

@author: Ulkesh
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD

import cv2, numpy as np
import pickle

from keras import backend as K
K.set_image_dim_ordering('th')
    
def hand_gesture(weights_path=None):
    
    img_width, img_height = 64, 64

    train_data_dir = 'train'
    validation_data_dir = 'val'
    nb_classes = 5
    nb_train_samples = 2000
    nb_validation_samples = 1000
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

    if weights_path:
        model.load_weights(weights_path)

    return model
    
def labelout (o):
    if o[0][0]==1.0:
        return 'Gesture-1'
    elif o[0][1]==1.0:
        return 'Gesture-2'
    elif o[0][2]==1.0:
        return 'Gesture-3'
    elif o[0][3]==1.0:
        return 'Gesture-4'
    elif o[0][4]==1.0:
        return 'Gesture-5'
        
        
        
if __name__ == "__main__":
#    the_filename='labelout.txt'
#    #with open(the_filename, 'wb') as f:
#    #    pickle.dump(my_list, f)
#    with open(the_filename, 'rb') as f:
#        my_list = pickle.load(f)
#    im = cv2.resize(cv2.imread('1_24.jpg'), (64, 64)).astype(np.float32)
#    im[:,:,0] -= 103.939
#    im[:,:,1] -= 116.779
#    im[:,:,2] -= 123.68
#    im = im.transpose((2,0,1))
#    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = hand_gesture('model2_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='nadam')
#    out = model.predict(im)
#    print (out)
#    print (labelout(out))
    #print np.argmax(out)
    #print my_list[np.argmax(out)]
    
    
    
    ########################
    # REAL-TIME PREDICTION #
    ########################

    print '... Initializing RGB stream'
    
     #### Initialize built-in webcam
    cap = cv2.VideoCapture(0)
    # Enforce size of frames
    cap.set(3, 320) 
    cap.set(4, 240)

    shot_id = 0
 
    #### Start video stream and online prediction
    while (True):
         # Capture frame-by-frame
    
#        start_time = time.clock()
        
        ret, frame = cap.read()
        
        #color_frame = color_stream.read_frame() ## VideoFrame object
        #color_frame_data = frame.get_buffer_as_uint8() ## Image buffer
        #frame = convert_frame(color_frame_data, np.uint8) ## Generate BGR frame
                
        im = cv2.resize(frame, (64, 64)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        
        out = model.predict(im)
        #print np.argmax(out)
        #print my_list[np.argmax(out)]
        
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        #r = 100.0 / frame.shape[1]
        dim = (640, 480)
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,labelout(out),(20,450), font, 1, (255,255,255),1,1)
        # Display the resulting frame
        cv2.imshow('DeepNN-ABB',resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()