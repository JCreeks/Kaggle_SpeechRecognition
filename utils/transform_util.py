#!/home/ubuntu/anaconda3/bin//python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 12/27/17
"""
import numpy as np
import pandas as pd

def sampleRate():
    new_sample_rate =16000
    print('\n sample rate: ', new_sample_rate, '\n')
    return new_sample_rate

def relabel():
    relabel = True
    print('\n relabel: ', relabel, '\n')
    return relabel

def pad_audio(samples, L=16000):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels, relabel=True, get_dummies=True):
    legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
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
    return nlabels

