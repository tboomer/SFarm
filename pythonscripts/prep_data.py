# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
np.random.seed(516)

import os
import glob
import cv2
import math
import pickle
# import datetime
# import pandas as pd
import sys

# from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import KFold
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
# from keras.models import model_from_json
# from keras.optimizers import Adadelta
# from sklearn.metrics import log_loss

# from sklearn.cross_validation import train_test_split


# Input image dimensions and color type
img_rows, img_cols = 48, 64
color_type = 1
# color_type = 1 - gray
# color_type = 3 - RGB

# For Tim (uncomment both lines below, comment Amit's line):
os.chdir('C:/Users/tboom_000/Documents/Personal/Projects/Kaggle/SFarm')
pth = '.'

# For Amit:
# pth = '/Users/amitshavit/Desktop/state_farm_kaggle_imgs'

def get_im(path, img_rows, img_cols, color_type):
    try:
        # Load image
        if color_type == 1 :
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path, 1)
        # Reduce size
        resized = cv2.resize(img, (img_cols, img_rows))
        return resized

    except:
        print('Failed reading', path)
        sys.exit(1)


def load_train(img_rows, img_cols, color_type):
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(pth,'.', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)

    return X_train, y_train


def load_test(img_rows, img_cols, color_type):
    print('Read test images')
    path = os.path.join(pth,'.', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data
    
def read_and_normalize_train_data(img_rows, img_cols, color_type):
    cache_path = os.path.join(pth,'.', 'cache', 'train_r_' + str(img_rows) + 
        '_c_' + str(img_cols) + "_" + str(color_type) + '.dat')
    if not os.path.isfile(cache_path):
        train_data, train_target = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target


def read_and_normalize_test_data(img_rows, img_cols, color_type):
    cache_path = os.path.join(pth,'.', 'cache', 'test_r_' + str(img_rows) + 
        '_c_' + str(img_cols) + "_" + str(color_type) + '.dat')
    if not os.path.isfile(cache_path):
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret
    
