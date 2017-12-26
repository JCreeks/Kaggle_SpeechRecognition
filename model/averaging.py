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

new_sample_rate = 8000
L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

from conf.configure import Configure

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

def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

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

def test_data_generator(batch=16):
    fpaths = glob(os.path.join(Configure.test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        imgs.append(specgram)
#         fnames.append(path.split('\\')[-1])
        fnames.append(path.split('/')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
        yield fnames, imgs
    raise StopIteration()

# labels, fnames = list_wavs_fname(Configure.train_data_path)

# new_sample_rate = 8000
# chopNum = 1000 #default num=20
# y_train = []
# #x_train = []

# for label, fname in zip(labels, fnames):
#     sample_rate, samples = wavfile.read(os.path.join(Configure.train_data_path, label, fname))
#     samples = pad_audio(samples)
#     if len(samples) > 16000:
#         n_samples = chop_audio(samples, num=chopNum)
#     else: n_samples = [samples]
#     for samples in n_samples:
#         resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
#         _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
#         y_train.append(label)
#         #x_train.append(specgram)
# #x_train = np.array(x_train)
# #x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
# y_train = label_transform(y_train, relabel=True, get_dummies=True)
# label_index = y_train.columns.values
# # y_train = y_train.values
# # y_train = np.array(y_train)
# del labels, fnames, y_train
# gc.collect()
#print(label_index)
label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up',\
 'yes']

# averaging modeling 
#weight = [] # default []
weight = [.5, .25, .25]
weight = [i/sum(weight) for i in weight]
index = []
results = []
modelPaths = glob(os.path.join(Configure.model_path, '*class12*'))
modelList = []
for modelFile in modelPaths:
    model = load_model(modelFile)
    modelList.append(model)
print("start averaging...")
for fnames, imgs in test_data_generator(batch=32):
    predList = []
    for model in modelList:
        predicts = model.predict(imgs)
        predList.append(predicts)
    if len(predList)==len(weight) and sum(weight)==1.0:
        predList = [a*b for a, b in zip(weight, predList)]
    predicts = sum(predList)/len(modelPaths)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
#     predicts = label_transform(predicts, relabel=True, get_dummies=False)
    index.extend(fnames)
    results.extend(predicts)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
# df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)
df.to_csv(Configure.submission_path, index=False)
