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

import time


class Configure(object):

    model_path = '../model'
    train_data_path = '../input/train/audio'
    test_data_path = '../input/test/audio'
    
    processed_train_path = '../input/train_dataset.pkl'
    processed_test_path = '../input/test_dataset.pkl'

    submission_path = '../output/submission_{}.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
