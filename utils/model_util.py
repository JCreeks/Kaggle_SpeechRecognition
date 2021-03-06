#!/home/ubuntu/anaconda3/bin//python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 1/3/18
"""

import os
import sys
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile

import tensorflow as tf
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure
from utils import data_util
from utils.transform_util import relabel, label_transform, pad_audio, chop_audio, sampleRate
# from utils.model_util import CNN

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


class CNN:
    
    def __init__(self):
        self.input_shape = (99,81,1)
        self.nclass = 12
        self.epochs = 5
        self.batch_size =18
        self.filtersList = [16,32,64,128]
        self.img_activation = 'relu'
        self.dense_activation = 'softmax'
        self.dropout_rate = 0.2
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.metrics=['accuracy']
        self.seed = 2018
        self.fitted_model = None
        self.kernel_size_0 = 2
        self.kernel_size_1 = 3
        self.kernel_size_2 = 3
        self.param_grid = {}

    def model(self, input_shape=(99,81,1), nclass=12, filtersList=[16,32,64,128],
              img_activation='relu', dense_activation='softmax', dropout_rate=0.2, 
              optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
              ):
        inp = Input(shape=input_shape)
        norm_inp = BatchNormalization()(inp)
        img_1 = Convolution2D(filtersList[0], kernel_size=self.kernel_size_0, activation=img_activation)(norm_inp)
        img_1 = Convolution2D(filtersList[0], kernel_size=self.kernel_size_0, activation=img_activation)(img_1)
        img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
        img_1 = Dropout(rate=dropout_rate)(img_1)
        img_1 = Convolution2D(filtersList[1], kernel_size=self.kernel_size_1, activation=img_activation)(img_1)
        img_1 = Convolution2D(filtersList[1], kernel_size=self.kernel_size_1, activation=img_activation)(img_1)
        img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
        img_1 = Dropout(rate=dropout_rate)(img_1)
        img_1 = Convolution2D(filtersList[2], kernel_size=self.kernel_size_2, activation=img_activation)(img_1)
        img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
        img_1 = Dropout(rate=dropout_rate)(img_1)
        img_1 = Flatten()(img_1)

        dense_1 = BatchNormalization()(Dense(filtersList[3], activation=img_activation)(img_1))
        dense_1 = BatchNormalization()(Dense(filtersList[3], activation=img_activation)(dense_1))
        dense_1 = Dense(nclass, activation=dense_activation)(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def randomTune(self, x_train, y_train, param_grid, cv=3, n_iter_search=10):
        # x_train, y_train = data_util.load_train()
        
        self.input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        self.nclass = y_train.shape[1]

        paramList = ['input_shape', 'nclass', 'epochs', 'filtersList', 'img_activation', 
                    'dense_activation', 'dropout_rate', 'optimizer', 'loss', 'metrics',
                    ]
        paramDict = {}
        for param in paramList:
            paramDict[param] = getattr(self,param) 
        
        model = KerasClassifier(build_fn=self.model(), verbose=0, 
                                epochs=self.epochs, batch_size=self.batch_size)

        np.random.seed(self.seed)
        # run randomized search
        random_search = RandomizedSearchCV(model, param_distributions=param_grid, verbose=0,
                                           n_iter=n_iter_search, n_jobs=-1, cv=cv)
        print('start searching')
        start = time()
        random_search.fit(x_train, y_train)
        self.fitted_model = random_search
        print("RandomizedSearchCV took %.2f mins for %d candidates"
              " parameter settings." % ((time() - start)/60, n_iter_search))
        self.report(random_search.cv_results_)

        print('done!')
        
    def report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
                
    def predict(self, x_test):
        model = self.fitted_model
        if (model == None):
            print('model not fitted yet!')
            return
        predicts = model.predict(x_test)
        return predicts
          
    def predict_save(self):
        relabel = relabel()
        x_test, test_fname = data_util.load_test()

        label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']
        print("start predicting...")
        predList = []
        model = self.fitted_model
        if (model == None):
            print('model not fitted yet!')
            return
        predicts = model.predict(x_test)
        predicts = np.argmax(predicts, axis=1)
        predicts = [label_index[p] for p in predicts]
        if not relabel:
            predicts = label_transform(predicts, relabel=True, get_dummies=False)

        df = pd.DataFrame(columns=['fname', 'label'])
        df['fname'] = test_fname
        df['label'] = predicts
        df.to_csv(Configure.submission_path, index=False)

        print('done!')
        
    def save_model(self):        
        new_sample_rate = sampleRate()
        modelName = 'sampleRate'+str(new_sample_rate)+'_nclass'+str(self.nclass)+'_seed'+str(self.seed)\
                    +'_epoch'+str(self.epoch)+'_CNN'+'.model'
        model = self.fitted_model
        if (model == None):
            print('model not fitted yet!')
            return
        model.save(os.path.join(Configure.model_path, modelName))
