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

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

relabel = True
L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

from conf.configure import Configure
from utils import data_util

def label_transform(labels, relabel=True, get_dummies=True):
    nlabels = []
    if relabel:
        for label in labels:
            if label == '_background_noise_':
                nlabels.append('silence')
            elif label not in legal_labels:
                nlabels.append('unknown')
            else:
                nlabels.append(label)
    else:
        nlabels = labels
    if get_dummies:
        return(pd.get_dummies(pd.Series(nlabels)))
    return labels

x_test, test_fname = data_util.load_test()

weight = []
#weight = [i/sum(weight) for i in weight]
label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']
modelPaths = glob(os.path.join(Configure.model_path, '*74*'))
modelList = []
for modelFile in modelPaths:
    model = load_model(modelFile)
    modelList.append(model)
print("start averaging...")
predList = []
for model in modelList:
    predicts = model.predict(x_test)
    predList.append(predicts)
if len(predList)==len(weight) and sum(weight)==1.0:
    predList = [a*b for a, b in zip(weight, predList)]
predicts = sum(predList)/len(modelPaths)
predicts = np.argmax(predicts, axis=1)
predicts = [label_index[p] for p in predicts]
if not relabel:
    predicts = label_transform(predicts, relabel=True, get_dummies=False)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = test_fname
df['label'] = predicts
df.to_csv(Configure.submission_path, index=False)
