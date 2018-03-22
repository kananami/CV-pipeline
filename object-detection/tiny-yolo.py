#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
# Copyright (c) 2017 - Limber Cheng <cheng@limberence.com>  
# @Time : 22/03/2018 16:52 
# @Author : Limber Cheng 
# @File : tiny-yolo 
# @Software: PyCharm 
from keras.layers import Convolution2D, LeakyReLU, MaxPooling2D, Flatten, Dense 
from keras.models import Sequential 
 
model = Sequential() 
model.add(Convolution2D(16, 3, 3, input_shape=(3, 448, 448), border_mode='same', subsample=(1, 1))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Convolution2D(32, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid')) 
model.add(Convolution2D(64, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid')) 
model.add(Convolution2D(128, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid')) 
model.add(Convolution2D(256, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid')) 
model.add(Convolution2D(512, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid')) 
model.add(Convolution2D(1024, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(Convolution2D(1024, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(Convolution2D(1024, 3, 3, border_mode='same')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(Flatten()) 
model.add(Dense(256)) 
model.add(Dense(4096)) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(Dense(1470)) 