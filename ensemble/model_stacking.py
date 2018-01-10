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
from utils.transform_util import label_transform, pad_audio, chop_audio, sampleRate, relabel

from stacking.model_wrapper import XgbWrapper, SklearnWrapper, GridCVWrapper
from stacking.model_stack import TwoLevelModelStacking

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score

def Accuracy(y_train, y_predict):
    return accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_predict, axis=1))

from keras import backend as K
import tensorflow as tf
n_jobs=100
config = tf.ConfigProto(intra_op_parallelism_threads=n_jobs, 
                        inter_op_parallelism_threads=n_jobs, 
                        allow_soft_placement=True, 
                        device_count = {'GPU': n_jobs})
session = tf.Session(config=config)
K.set_session(session)

import time
from xgboost import XGBClassifier

relabel = relabel()
sample_rate = sampleRate()
label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']

x_train, y_train = data_util.load_train()
x_test, test_fname = data_util.load_test()

# n=500
# index = np.random.choice(x_train.shape[0], n, replace=False)
# x_train=x_train[index,:,:,:]
# y_train=y_train[index,:]
# x_test=x_test[index,:,:,:]
# test_fname=pd.Series(test_fname).iloc[index]

gc.collect()

modelPaths = glob(os.path.join(Configure.model_path, '*16000*'))
modelList = []
for modelFile in modelPaths:
    model = load_model(modelFile)
    modelList.append(model)
    
#stacking_model=XgbWrapper()
stacking_model = XGBClassifier()
#stacking_model=LogisticRegression()

model_stack = TwoLevelModelStacking(x_train, y_train, x_test, modelList, stacking_model=stacking_model, stacking_with_pre_features=False, n_folds=3, random_seed=0, 
                                    scorer = Accuracy)
print("start stacking...")

predicts = model_stack.run_stack_predict()
# print(predicts)

cmd = 'rm ../output/stacking*.csv'
os.system(cmd)  
out = pd.DataFrame(predicts)
out.columns = label_index
out['fname'] = test_fname
submission_path = '../output/stacking_added_{}.csv'.format(time.strftime('%Y_%m_%d', time.localtime(time.time())))
out.to_csv(submission_path, index=False)

predicts = np.argmax(predicts, axis=1)
predicts = [label_index[p] for p in predicts]
if not relabel:
    predicts = label_transform(predicts, relabel=True, get_dummies=False)

cmd = 'rm ../output/subm*.csv'
os.system(cmd)    
df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = test_fname
df['label'] = predicts
df.to_csv(Configure.submission_path, index=False)

print('done!')
