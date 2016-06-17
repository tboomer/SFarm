# Obtained from this thread: https://www.kaggle.com/c/state-farm-distracted-
# driver-detection/forums/t/19971/simple-solution-keras as further modified
# by Amit Shavit

#

import numpy as np
np.random.seed(516)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import sys

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import Adadelta
from sklearn.metrics import log_loss

from sklearn.cross_validation import train_test_split

# Input image dimensions and color type
img_rows, img_cols = 48, 64
color_type = 1
# color_type = 1 - gray
# color_type = 3 - RGB

# For Tim (uncomment both lines below, comment Amit's line):
os.chdir('C:/Users/tboom_000/Documents/Personal/Projects/Kaggle/StateFarm')
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


# def save_model(model):
#    json_string = model.to_json()
#    if not os.path.isdir('cache'):
#        os.mkdir('cache')
#    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
#    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)
#
#def read_model():
#    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
#    model.load_weights(os.path.join('cache', 'model_weights.h5'))
#    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def create_submission(predictions, test_id, loss, **kwargs):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('submissions'):
        os.mkdir('submissions')

    #suffix = str(loss) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '_'
    suffix = '%1.3f_%s' % (loss, str(now.strftime("%Y-%m-%d-%H-%M")))

    for arg in kwargs:
        suffix += '_' + str(arg) + '=' + str(kwargs[arg]).replace(', ','x').replace('(','').replace(')','')

    sub_file = os.path.join('submissions', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data(img_rows, img_cols, color_type):
    cache_path = os.path.join(pth,'.', 'cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + "_" + str(color_type) + '.dat')
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
    cache_path = os.path.join(pth,'.', 'cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + "_" + str(color_type) + '.dat')
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


def merge_several_folds_fast(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

"""Don't use for now...
def run_cross_validation(nfolds=10):

    # Neural net parameters
    batch_size = 64
    nb_classes = 10
    nb_epoch = 3
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    random_state = 51

    train_data, train_target = read_and_normalize_train_data(img_rows, img_cols, color_type)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, test_index in kf:
        num_fold += 1
        print('Start KFold number {} of {}'.format(num_fold, nfolds))
        X_train, X_valid = train_data[train_index], train_data[test_index]
        Y_train, Y_valid = train_target[train_index], train_target[test_index]
        print('Split train: ', len(X_train))
        print('Split valid: ', len(X_valid))

        model = Sequential()
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(color_type, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128 * color_type)) # from 128
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        adadel = Adadelta(lr = 0.01)
        model.compile(loss='categorical_crossentropy', optimizer= adadel, metrics=['accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_valid, Y_valid))

        # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
        # print('Score log_loss: ', score[0])

        predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        yfull_test.append(test_prediction)

    score = log_loss(train_target, dict_to_list(yfull_train))
    print('Final score log_loss: ', score)

    test_res = merge_several_folds_fast(yfull_test, nfolds)
    create_submission(test_res, test_id, score)
    print(model.summary())

# run_cross_validation(3)
"""

def get_drivers():
    # read drivers
    print('Reading drivers...')
    drivers = pd.read_csv(os.path.join(pth,'.','input','driver_imgs_list.csv'))
    filename_for_row_in_train = [] # assumes two drivers cannot have the same filename and same filename cannot be in two classes

    for j in range(10):
        path = os.path.join(pth,'.', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for filename in files:
            filename_for_row_in_train.append(os.path.basename(filename))

    train_drivers = pd.DataFrame(filename_for_row_in_train,columns=['filename'])
    train_drivers = train_drivers.merge(drivers, left_on='filename', right_on='img', how='left')

    return train_drivers

def run():

    # Neural net parameters
    batch_size = 68
    nb_classes = 10
    nb_epoch = 1
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # Define learning rate to Adadelta
    lr = 1.0

    train_data, train_target = read_and_normalize_train_data(img_rows, img_cols, color_type)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type)

    # Get the drivers
    train_drivers_per_train_row = get_drivers()

    # Split training set into train/validation by drivers.
    # Validation set consists of all images for one driver.
    val_driver = train_drivers_per_train_row.subject.unique()[0]
    val_idx = train_drivers_per_train_row[train_drivers_per_train_row.subject == val_driver].index
    train_idx = train_drivers_per_train_row[~(train_drivers_per_train_row.subject == val_driver)].index

    X_train, X_val, y_train, y_val = (train_data[train_idx,:],
        train_data[val_idx,:], train_target[train_idx], train_target[val_idx])

    # Split the data into training/validation sets (OLD)
    #X_train, X_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.2, random_state=0)

    print('Begin training...')

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128 * color_type)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    adadelta = Adadelta(lr=lr, rho=0.95, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer= adadelta, metrics=['accuracy'])

    print(model.summary())

    print('Training...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
    validation_data=(X_val,y_val))

    print('Predicting on validation set...')
    y_pred = model.predict(X_val, batch_size=batch_size, verbose=1)
    score = log_loss(y_val, y_pred)
    print('Score log_loss: ', score)

    print('Predicting on test set...')

    # Predict on test set
    y_test_pred = model.predict(test_data, batch_size=batch_size, verbose=1)
    create_submission(y_test_pred, test_id, score, nepoch=nb_epoch,
                      dim=(img_rows,img_cols),learnrate=lr)


run()

#-----------------------------------------------------------------------------
# Save results for analysis in R
np.savetxt('./cache/predictions.csv', y_pred, delimiter=',')
np.savetxt('./cache/yvals.csv', y_val, delimiter=',')
