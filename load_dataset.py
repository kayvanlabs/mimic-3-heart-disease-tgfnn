# -*- coding: utf-8 -*-

import sklearn.datasets
import pandas as pd
import numpy as np


def load_data(dir):
    # load data
    X_train = pd.read_csv(f'{dir}/x_train.csv')
    y_train = pd.read_csv(f'{dir}/y_train.csv')
    X_test = pd.read_csv(f'{dir}/x_test.csv')
    y_test = pd.read_csv(f'{dir}/y_test.csv')

    feature_names = X_train.columns.tolist()

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.squeeze()
    y_test = y_test.values.squeeze()

    # determine each features' category
    category_info = np.zeros(X_train.shape[1]).astype(int) # initialize to zeros
    for i in range(X_train.shape[1]):
        if list(np.unique(X_train[:,i])) == [0,1]:
            category_info[i] = 2

    # compute class weights
    classes, counts = np.unique(y_train, return_counts=True)
    weights = counts.sum() / (2 * counts)
    weights_dict = dict(zip(classes, weights))

    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'num_classes': len(np.unique(y_train)),
        'category_info': category_info,
        'feature_names': feature_names,
        'weights': weights_dict
    }

    return dataset
