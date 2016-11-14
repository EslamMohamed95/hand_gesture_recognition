import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Convolution2D, Dropout, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
import cv2

# Layers Convolution size
CONV_1_SIZE = 5
CONV_2_SIZE = 5
CONV_3_SIZE = 5
CONV_4_SIZE = 5
CONV_5_SIZE = 3
CONV_6_SIZE = 3
 
def create_model(input_shape,output_shape):

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
    
    dense_2 = Dense(128, name='dense_2')(dropout_4)
    batch_normalization_8 = BatchNormalization(name='batch_normalization_8')(dense_2)
    relu_8 = Activation('relu')(batch_normalization_8)
    dropout_5 = Dropout(0.2)(relu_8)
    
    out = Dense(output_shape, activation='softmax', name='Output')(dropout_5)
    
    # Creating a model from the specified layers
    model = Model(input = inp, output = out)
    
    model.load_weights('hand_gesture_weights_36.h5', by_name=True)
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":   
    ########################
    # REAL-TIME PREDICTION #
    ########################

    print '... Initializing RGB stream'
    
     #### Initialize built-in webcam
    cap = cv2.VideoCapture(0)
    # Enforce size of frames
    cap.set(3, 320) 
    cap.set(4, 240)
    model = create_model((1, 64, 64),6)
    shot_id = 0
 
    #### Start video stream and online prediction
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = (np.array(cv2.resize(im, (64, 64))))        
        im = np.array(im).reshape(1, 1, 64, 64)
    
        
        out = model.predict(im)
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        #r = 100.0 / frame.shape[1]
        dim = (640, 480)
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,'{}'.format(np.argmax(out)),(20,450), font, 1, (255,255,255),1,1)
        # Display the resulting frame
        cv2.imshow('Gesture Recognition',resized)
        cv2.waitKey(30)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
    # When everything done, release the capture
            
    cap.release()
    cv2.destroyAllWindows()

 