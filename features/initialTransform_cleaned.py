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
import pickle

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

from conf.configure import Configure
from utils import data_util
from utils.transform_util import relabel, label_transform, pad_audio, chop_audio, sampleRate


# Here are custom_fft and log_specgram functions written by __DavidS__.


def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# Following is the utility function to grab all wav files inside train data folder.

with open(Configure.outlierNameList, "rb") as f:
    outlierNameList = pickle.load(f)
print("outlier num: ", len(outlierNameList))

def list_wavs_fname2(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        filename = fpath.split(dirpath+'/')[1]
        if r and not filename in outlierNameList:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        filename = fpath.split(dirpath+'/')[1]
        if r and not filename in outlierNameList:
            fnames.append(r.group(1))
    return labels, fnames



# __pad_audio__ will pad audios that are less than 16000(1 second) with 0s to make them all have the same length.
# 
# __chop_audio__ will chop audios that are larger than 16000(eg. wav files in background noises folder) to 16000 in length. In addition, it will create several chunks out of one large wav files given the parameter 'num'.
# 
# __label_transform__ transform labels into dummies values. It's used in combination with softmax to predict the label.

# def pad_audio(samples):
#     if len(samples) >= L: return samples
#     else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

# def chop_audio(samples, L=16000, num=20):
#     for i in range(num):
#         beg = np.random.randint(0, len(samples) - L)
#         yield samples[beg: beg + L]

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
#     return nlabels


# Next, we use functions declared above to generate x_train and y_train.
# label_index is the index used by pandas to create dummy values, we need to save it for later use.

labels, fnames = list_wavs_fname2(Configure.train_data_path)

relabel = relabel()
new_sample_rate = sampleRate() #8000
chopNum = 1000 #default num=20
y_train = []
x_train = []

for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(Configure.train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples, num=chopNum)
    else: n_samples = [samples]
    for samples in n_samples:
        if new_sample_rate>=sample_rate:
            resampled = samples
        else:
            resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)
x_train = np.array(x_train)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
y_train = label_transform(y_train, relabel=relabel, get_dummies=True)


label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
del labels, fnames
gc.collect()

print('x_train:', x_train.shape, ', y_train:', y_train.shape)
print("Save train data...")
data_util.save_cleaned_dataset(x_train, y_train)

del x_train, y_train
gc.collect()

print("done saving cleaned data!")
