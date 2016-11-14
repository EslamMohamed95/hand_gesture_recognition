# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:46:21 2016

@author: syamprasadkr
"""

import os
from random import shuffle
from shutil import copyfile

SRC_PATH = 'data_unsorted/'
DES_PATH1 = 'data/train/'
DES_PATH2 = 'data/val/'
DES_PATH3 = 'data/test/'
files = os.listdir(SRC_PATH)
shuffle(files)
shuffle(files)

num_files = len(files)
#print files
i=0
train_lim = int(0.5*num_files)
val_lim = train_lim + int(0.25*num_files)

for fl in files:
    source_path = os.path.join(SRC_PATH,fl)
    if i < train_lim:
        des_path = os.path.join(DES_PATH1,fl[:1])
        dest_path = os.path.join(des_path,fl)
        copyfile(source_path,dest_path)
    elif i >= train_lim and i < val_lim:
        des_path = os.path.join(DES_PATH2,fl[:1])
        dest_path = os.path.join(des_path,fl)
        copyfile(source_path,dest_path)
    else:
        des_path = os.path.join(DES_PATH3,fl[:1])
        dest_path = os.path.join(des_path,fl)
        copyfile(source_path,dest_path)
    i = i + 1
         