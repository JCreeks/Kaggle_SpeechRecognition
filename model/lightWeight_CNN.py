
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

relabel = True

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

from conf.configure import Configure
#src folders
# root_path = r'..'
# out_path = os.path.join(root_path, 'output')
# model_path = r'.'
# train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
# test_data_path = os.path.join(root_path, 'input', 'test', 'audio')


# Here are custom_fft and log_specgram functions written by __DavidS__.

# In[3]:


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

# In[4]:


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


# __pad_audio__ will pad audios that are less than 16000(1 second) with 0s to make them all have the same length.
# 
# __chop_audio__ will chop audios that are larger than 16000(eg. wav files in background noises folder) to 16000 in length. 
# In addition, it will create several chunks out of one large wav files given the parameter 'num'.
# 
# __label_transform__ transform labels into dummies values. It's used in combination with softmax to predict the label.

# In[5]:


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


# Next, we use functions declared above to generate x_train and y_train.
# label_index is the index used by pandas to create dummy values, we need to save it for later use.

# In[6]:


labels, fnames = list_wavs_fname(Configure.train_data_path)

new_sample_rate = 8000
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
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)
x_train = np.array(x_train)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
if relabel:
    y_train = label_transform(y_train, relabel=True, get_dummies=True)
else:
    y_train = label_transform(y_train, relabel=False, get_dummies=True)

label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
del labels, fnames
gc.collect()


# CNN declared below.
# The specgram created will be of shape (99, 81), but in order to fit into Conv2D layer, we need to reshape it.

# In[8]:


input_shape = (99, 81, 1)
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


# Test data is way too large to fit in RAM, we need to process them one by one.
# Generator test_data_generator will create batches of test wav files to feed into CNN.

# In[9]:


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


# We use the trained model to predict the test data's labels.
# However, since Kaggle doesn't provide test data, the following sections won't be executed here.

# In[10]:


# from keras.models import load_model
# model = load_model('cnn.model')


# In[13]:


#exit() #delete this
del x_train, y_train
gc.collect()


# In[14]:


index = []
results = []
for fnames, imgs in test_data_generator(batch=32):
    predicts = model.predict(imgs)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
    #predicts = label_transform(predicts, relabel=True, get_dummies=False)
    index.extend(fnames)
    results.extend(predicts)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
# df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)
df.to_csv(Configure.submission_path, index=False)


# In[ ]:


# averaging modeling 
index = []
results = []
modelPaths = glob(os.path.join(Configure.model_path, '*class12*'))
for fnames, imgs in test_data_generator(batch=32):
    predList = [load_model(model).predict(imgs) for model in modelPaths]
    predicts = sum(predList)/len(modelPaths)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
    if not relabel:
        predicts = label_transform(predicts, relabel=True, get_dummies=False)
    index.extend(fnames)
    results.extend(predicts)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
# df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)
df.to_csv(Configure.submission_path, index=False)

