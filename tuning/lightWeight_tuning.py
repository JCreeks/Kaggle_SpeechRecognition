
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
            
x_train, y_train = data_util.load_train()

g = CNN()
g.epochs = 10
g.batch_size = 20

batch_size = [10, 20, 40]
epochs = [5, 10]
# dropout_rate = [.1,.2,.3,.4]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# run randomized search
n_iter_search = 3
cv = 3
g.randomTune(x_train, y_train, param_grid=param_grid, cv=cv, n_iter_search=n_iter_search)
