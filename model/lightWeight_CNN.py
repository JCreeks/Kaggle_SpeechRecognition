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

from keras import backend as K
import tensorflow as tf

n_jobs=50
config = tf.ConfigProto(intra_op_parallelism_threads=n_jobs, 
                        inter_op_parallelism_threads=n_jobs, 
                        allow_soft_placement=True, 
                        device_count = {'GPU': n_jobs})
session = tf.Session(config=config)
K.set_session(session)

from conf.configure import Configure
from utils import data_util
from utils.transform_util import relabel, label_transform, pad_audio, chop_audio, sampleRate

relabel = relabel()
new_sample_rate = sampleRate()#8000
chopNum = 1000
L = 16000
hyper_pwr = 0.3
filtersList = [16,32,64] #default [8,16,32]
seed = 2018
np.random.seed(seed)
epoch = 10 #default 3
batch_size = 20
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']

# Function to compute class weights
def comp_cls_wts(y, pwr = 0.2):
    '''
    Used to compute class weights
    '''
    labelCount = y.sum(axis=0)
    dic = {}
    for i in range(y.shape[1]):
#         print(labelCount[i])
        dic[i] = labelCount[i]**pwr/np.sum(labelCount)**pwr
    return dic

x_train, y_train = data_util.load_train()

cls_wts = comp_cls_wts(y_train, pwr = hyper_pwr)
print(cls_wts)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
nclass = y_train.shape[1]#12

inp = Input(shape=input_shape)
norm_inp = BatchNormalization()(inp)

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
model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid), epochs=epoch, 
          class_weight = cls_wts, shuffle=True, verbose=2)

modelName = 'sampleRate'+str(new_sample_rate)+'_nclass'+str(nclass)+'_seed'+str(seed)+'_batchSize'+str(batch_size)            +'_epoch'+str(epoch)+'_ConvDouble'+'.model'
model.save(os.path.join(Configure.model_path, modelName))

del x_train, x_valid, y_train, y_valid
gc.collect()

x_test, test_fname = data_util.load_test()

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
