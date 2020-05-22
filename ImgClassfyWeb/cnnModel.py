import tensorflow as tf
from tensorflow.keras.layers import  Dense,\
        Conv2D , MaxPooling2D,BatchNormalization,\
        Flatten,Dropout

import numpy as np
import getConfig

gConfig = getConfig.get_config()

class cnnModel(object):
    def __init__(self , rate):

        # dropout rate
        self.rate  =rate

    def create_model(self):

        model  = tf.keras.Sequential()

        model.add(
                Conv2D(
                    32,
                    (3,3),
                    kernel_initializer = 'he_normal',
                    strides=1,
                    padding='same',
                    activation = 'relu',
                    input_shape =[32,32,3]))
        model.add(
                MaxPooling2D(
                    (2,2),
                    strides = 1,
                    padding = 'same'))
        model.add(BatchNormalization())
        model.add(
                Conv2D(
                    64,
                    (3,3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    activation='relu'))
        model.add(
                MaxPooling2D(
                    (2,3),
                    strides=1,
                    padding='same'))
        model.add(BatchNormalization())
        model.add(
                Conv2D(
                    128,
                    (3,3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    activation='relu'))
        model.add(
                MaxPooling2D(
                    (2,2),
                    strides=1,
                    padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(rate = self.rate))
        model.add(Dense(10,activation='softmax'))

        model.compile(
                loss='categorical_crossentropy',
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ['accuracy'])
        return model
