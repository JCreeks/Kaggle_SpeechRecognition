
# coding: utf-8


# In[1]:


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



from conf.configure import Configure
from utils import data_util
from utils.transform_util import relabel, label_transform, pad_audio, chop_audio, sampleRate

relabel = relabel()
new_sample_rate = sampleRate()#8000
chopNum = 1000
L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

# def label_transform(labels, relabel=True, get_dummies=True):
#     nlabels = []
#     if relabel:
#         for label in labels:
#             if label == '_background_noise_':
#                 nlabels.append('silence')
#             elif label not in legal_labels:
#                 nlabels.append('unknown')
#             else:
#                 nlabels.append(label)
#     else:
#         nlabels = labels
#     if get_dummies:
#         return(pd.get_dummies(pd.Series(nlabels)))
#     return labels

x_train, y_train = data_util.load_train()

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
nclass = y_train.shape[1]#12
seed = 2017
inp = Input(shape=input_shape)
norm_inp = BatchNormalization()(inp)
filtersList = [16,32,64] #default [8,16,32]
epoch = 5 #default 3
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
model.summary()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
model.fit(x_train, y_train, batch_size=16, validation_data=(x_valid, y_valid), epochs=epoch, shuffle=True, verbose=2)

modelName = 'sampleRate'+str(new_sample_rate)+'_nclass'+str(nclass)+'_seed'+str(seed)+'_chopNum'+str(chopNum)            +'_epoch'+str(epoch)+'_ConvDouble'+'.model'
model.save(os.path.join(Configure.model_path, modelName))

del x_train, x_valid, y_train, y_valid
gc.collect()

x_test, test_fname = data_util.load_test()

label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']
print("start predicting...")
predList = []
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