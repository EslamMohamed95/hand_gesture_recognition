# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:19:23 2016

@author: syamprasadkr
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import Adam

import cv2, numpy as np
#import pickle
from keras import backend as K
K.set_image_dim_ordering('th')
    
def hgm(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape = (1, 64, 64)))
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

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
#    the_filename='labelout.txt'
    #with open(the_filename, 'wb') as f:
    #    pickle.dump(my_list, f)
#    with open(the_filename, 'rb') as f:
    my_list = ['Gesture 1', 'Gesture 2', 'Gesture 3', 'Gesture 4', 'Gesture 5']
#    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#    im[:,:,0] -= 103.939
#    im[:,:,1] -= 116.779
#    im[:,:,2] -= 123.68
#    im = im.transpose((2,0,1))
#    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = hgm('3_hgm.h5')
#    f = open('model_summary.dat', 'w+')
#    
#    f.write(model.summary())
#    f.close()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#    out = model.predict(im)
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
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.array(im).reshape(1, 1, 64, 64)
#        im[:,:,0] -= 103.939
#        im[:,:,1] -= 116.779
#        im[:,:,2] -= 123.68
#        im = im.transpose((2,0,1))
#        im = np.expand_dims(im, axis=0)
        
        out = model.predict(im)
        #print np.argmax(out)
        #print my_list[np.argmax(out)]
        
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        #r = 100.0 / frame.shape[1]
        dim = (352, 240)
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,my_list[np.argmax(out)],(20,200), font, 1, (255,255,255),1,1)
        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition',resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()