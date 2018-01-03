
# coding: utf-8

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
from utils.model_util import CNN

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

seed = 2018
np.random.seed(seed)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
x_train, y_train = data_util.load_train()

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
nclass = y_train.shape[1]
model = KerasClassifier(build_fn=CNN(input_shape, nclass), verbose=0)

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [5, 10, 20, 50]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(model, param_distributions=param_grid,
                                   n_iter=n_iter_search, n_jobs=-1, cv=3)
print('start searching')
start = time()
random_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f mins for %d candidates"
      " parameter settings." % ((time() - start)/60, n_iter_search))
report(random_search.cv_results_)

print('done!')