#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 12/26/17
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# import cPickle
import pickle
import pandas as pd
from conf.configure import Configure


def load_dataset():
    if not os.path.exists(Configure.processed_train_path):
        train = pd.read_csv(Configure.original_train_path)
    else:
        with open(Configure.x_train_path, "rb") as f:
            x_train = pickle.load(f)
#             x_train = cPickle.load(f)
        with open(Configure.y_train_path, "rb") as f:
            y_train = pickle.load(f)
#             y_train = cPickle.load(f)

    if not os.path.exists(Configure.processed_test_path):
        test = pd.read_csv(Configure.original_test_path)
    else:
        with open(Configure.x_test_path, "rb") as f:
            x_test = pickle.load(f)
#             x_test = cPickle.load(f)
    return x_train, y_train, x_test


def save_dataset(x_train, y_train, x_test):
    if x_train is not None:
        with open(Configure.x_train_path, "wb") as f:
            pickle.dump(x_train, f, -1)
#             cPickle.dump(x_train, f, -1)
            
    if y_train is not None:
        with open(Configure.y_train_path, "wb") as f:
            pickle.dump(y_train, f, -1)
#             cPickle.dump(y_train, f, -1)

    if x_test is not None:
        with open(Configure.x_test_path, "wb") as f:
            pickle.dump(x_test, f, -1)
#             cPickle.dump(x_test, f, -1)
