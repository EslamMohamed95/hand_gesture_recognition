# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:48:13 2016

@author: syamprasadkr
"""
import numpy as np
import cv2
import os
from random import shuffle

PATH_BASE = ['Gesture_1', 'Gesture_2', 'Gesture_3', 'Gesture_4', 'Gesture_5']
PATH_TARGET = 'data_unsorted/'
for folder in PATH_BASE:
    path = os.path.join('dataset/',folder)
    #print path
    files=os.listdir(path)
    shuffle(files)
    #print files
    i=0
    for fl in files:
        fname = os.path.join(path,fl)
#        print fname
        pre,ext = os.path.splitext(fl)
        #print ext
        if ext != '.jpg':
            ext = '.jpg'
            
        if pre.isdigit() or pre[:2] == 'na' or pre[:4] == 'What' or pre[:3] == 'acd' :
            continue
#        print (fname)
        try:
            img = cv2.imread(fname,0)
            img = cv2.resize (img,(64,64))
        except:
            continue
#        cv2.imshow('image',img)
#        cv2.waitKey(100)
#        cv2.destroyAllWindows()
        if pre[:2] == 'aj' or pre[:2] == 'mb' or pre[:3] == 'RAM' or pre[:2] == 'sd' or pre[:3] =='sng':
            rows,cols = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
            img = cv2.warpAffine(img,M,(cols,rows))
#            cv2.imshow('image',img)
#            cv2.waitKey(100)
#            cv2.destroyAllWindows()
            
        if pre[:1] == 'g' and pre[2:4] == '-b':
            rows,cols = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
            img = cv2.warpAffine(img,M,(cols,rows))
#            cv2.imshow('image',img)
#            cv2.waitKey(100)
#            cv2.destroyAllWindows()
            
        ftarget = PATH_TARGET + '/' +folder[-1]+'_'+str(i)+ext
        cv2.imwrite(ftarget,img)
        i = i+1    
         
        
            
        