#!/home/ubuntu/anaconda3/bin//python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 1/3/18
"""

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model

def CNN(input_shape, nclass):
    # create model
#     input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
#     nclass = y_train.shape[1]#12
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    filtersList = [16,32,64] #default [8,16,32]
#     epoch = 5 #default 3
    img_1 = Convolution2D(filtersList[0], kernel_size=2, activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(filtersList[0], kernel_size=2, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(filtersList[1], kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(filtersList[1], kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(filtersList[2], kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.binary_crossentropy)
    return model