# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:54:15 2016

@author: tboom_000
"""

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