# -*- coding: utf-8 -*-
"""
Load public datasets and synthetic datasets for classification.

The data loading functions will return a dictionary with the following keys:
variables: A np.ndarray of features with a shape (the number of samples, the number of features)
response: A np.ndarray of labels with a shape (the number of samples,)
num_classes: Int. The number of classes
category_info: A np.array with a shape (the number of features).  entry category_info[i] = 0 means the i-th variable 
        is a continuous variable. entry category_info[i] > 0 means the i-th variable is a categorical variable, and 
        the value is the number of levels. This information is only used in the proposed machine learning technique.

Created on Wed Dec  2 13:35:59 2020

@author: hemingy
"""

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


def load_custom_data(input_file):

    # get data
    df = pd.read_csv(input_file)

    # split
    X_train = df[df['TRAIN'] == 1].drop(['TRAIN', 'CASE'], axis=1).values
    y_train = df[df['TRAIN'] == 1]['CASE'].values
    X_test = df[df['TRAIN'] == 0].drop(['TRAIN', 'CASE'], axis=1).values
    y_test = df[df['TRAIN'] == 0]['CASE'].values

    # determine each features' category
    category_info = np.zeros(X_train.shape[1]).astype(int) # initialize to zeros
    for i in range(X_train.shape[1]):
        if list(np.unique(X_train[:,i])) == [0,1]:
            category_info[i] = 2

    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'num_classes': 2,
        'category_info': category_info,
        'feature_names': df.drop(['TRAIN', 'CASE'], axis=1).columns
    } 
    return dataset


def load_breast_data():
    data = sklearn.datasets.load_breast_cancer()
    feature_names = list(data.feature_names)
    X, y = pd.DataFrame(data.data, columns=feature_names), data.target
    
    dataset = {
        'variables': X,
        'response': y,
        'num_classes': 2,
        'category_info': np.zeros(X.shape[1]).astype(np.int)
    }
    return dataset


def load_iris_data():
    data = sklearn.datasets.load_iris()
    feature_names = list(data.feature_names)
    X, y = pd.DataFrame(data.data, columns=feature_names), data.target
    
    dataset = {
        'variables': X,
        'response': y,
        'num_classes': 3,
        'category_info': np.zeros(X.shape[1]).astype(np.int)
    }
    return dataset


def load_digits_data():
    data = sklearn.datasets.load_digits()
    feature_names = list(data.feature_names)
    X, y = pd.DataFrame(data.data, columns=feature_names), data.target
    
    dataset = {
        'variables': X,
        'response': y,
        'num_classes': 10,
        'category_info': np.zeros(X.shape[1]).astype(np.int)
    }
    return dataset


def load_wine_data():
    data = sklearn.datasets.load_wine()
    feature_names = list(data.feature_names)
    X, y = pd.DataFrame(data.data, columns=feature_names), data.target
    
    dataset = {
        'variables': X,
        'response': y,
        'num_classes': 3,
        'category_info': np.zeros(X.shape[1]).astype(np.int)
    }
    return dataset


def load_heart_data():
    # https://www.kaggle.com/ronitf/heart-disease-uci
    df = pd.read_csv('./dataset/heart.csv')
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]

    feature_names = X_df.columns
    category = pd.Series(np.zeros([len(feature_names)]), feature_names)
    category.sex = 2
    category.fbs = 2
    category.exang = 2
    category.thal = 4
    category_info = np.array(category)
    
    dataset = {
        'variables': X_df.to_numpy(),
        'response': y_df.to_numpy(),
        'num_classes': 2,
        'category_info': category_info
    }
    return dataset
    

def generate_gaussian_data_categorical(n_samples, mislabel=None, random_state=None):
    """ Synthetic dataset 1

    The dataset is simulated by sampling features from Gaussian distributions and assigning labels
    using randomly generated rules.
    
    Parameters
    ---------- 
    n_samples: Int. The number of data samples that should be generated.
    mislabel: Float. The ratio of data samples that is mis-labeled. (To add noises to the dataset)
    radom_state: Int. Random state.
    """
    np.random.seed(random_state)
    
    x0 = 2**0.5*np.random.randn(n_samples, 1)
    x1 = 3**0.5*np.random.randn(n_samples, 1) + 5
    x2 = 5**0.5*np.random.randn(n_samples, 1) - 1
    x3 = 2**0.5*np.random.randn(n_samples, 1) + 1
    x4 = np.random.randn(n_samples, 1) - 2
    x5 = np.expand_dims(np.random.randint(2, size=n_samples), axis=-1)
    x_noise = np.random.randn(n_samples, 2)
    features = np.concatenate([x0, x1, x2, x3, x4, x5, x_noise], axis=-1)
    
    # Generate labels:    
    rules = [np.logical_and.reduce([x1<3.8, x2>-2, x5==1], axis=0),
              np.logical_and.reduce([x1>6.3, x2>-2, x5==1], axis=0),
              np.logical_and.reduce([x0<1, x3>2, x5==0], axis=0),
              np.logical_and.reduce([x2>0, x4>-1, x5==0], axis=0),
              np.logical_and.reduce([x0<1, x4>-1.5, x5==0], axis=0)]
    
    for i in range(len(rules)):
        print('Rule {}: {:.2f}%'.format(i, np.sum(rules[i])/n_samples*100))
    
    labels = np.logical_or.reduce(rules, axis=0)[:,0]
    
    if mislabel is not None:
        one_array = labels[labels==1]
        mutated_one_array = np.where(np.random.random(one_array.shape) < mislabel, False, one_array)
        labels[labels==1] = mutated_one_array
    
    
    categorical_info = np.zeros([features.shape[1]]) 
    categorical_info[5] = 2
    # Add random noise
    features[:, categorical_info==0] += np.random.randn(features[:, categorical_info==0].shape[0], 
                                                        features[:, categorical_info==0].shape[1])*0.01
    print('Positive samples: {:.2f}%'.format(np.sum(labels)/features.shape[0]*100))
    
    
    rule_data =  [
                      {'Relation': [[1, 0], [2, 1], [2, 2], [5,1]], #[0,0], [0,1], 
                                'Out_weight': 1},
                      {'Relation': [[1, 2], [2, 1], [2, 2], [5,1]], #[0,0], [0,1], 
                                'Out_weight': 1},
                      {'Relation': [[0,0], [3,2], [5,0]],
                                'Out_weight': 1},
                      {'Relation': [[2,2], [4,2], [5,0]],
                                'Out_weight': 1}, 
                      {'Relation': [[0,0], [4,1], [4,2], [5,0]],
                                'Out_weight': 1}, 
                    ]
    
    # partially correct
    # rule_data =  [
    #                  {'Relation': [[1, 0],  [5,1]], #[0,0], [0,1], 
    #                            'Out_weight': 1},
                     
    #                  {'Relation': [[0,0], [4,0], [6,1]],
    #                            'Out_weight': 1},

    #                  {'Relation': [[0,0], [4,2], [6,1], [5,0]],
    #                            'Out_weight': 1}, 
    #                 ]
    
    
    dataset = {
        'variables': features.astype(np.float32),
        'response': labels.astype(np.int8),
        'num_classes': 2,
        'category_info': categorical_info.astype(np.int8),
        'rule_data': rule_data,
        'feature_names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
        }
    
    np.random.seed(None)
    return dataset


def generate_gaussian_data_complicated_function(n_samples, mislabel=None, random_state=None):
    """ Synthetic dataset 2

    The dataset is simulated by sampling features from Gaussian distributions and assigning labels 
    using a randomaly generated complicated function.

    Parameters
    ---------- 
    n_samples: Int. The number of data samples that should be generated.
    mislabel: Float. The ratio of data samples that is mis-labeled. (To add noises to the dataset)
    radom_state: Int. Random state.
    """

    np.random.seed(random_state)
    
    x0 = 2**0.5*np.random.randn(n_samples, 1)
    x1 = 3**0.5*np.random.randn(n_samples, 1) + 5
    x2 = 5**0.5*np.random.randn(n_samples, 1) - 1
    x3 = 2**0.5*np.random.randn(n_samples, 1) + 1
    x4 = np.random.randn(n_samples, 1) - 2
    x5 = 4.4**0.5*np.random.randn(n_samples, 1) - 1
    x6 = 1.2**0.5*np.random.randn(n_samples, 1)
    
    x_noise = np.random.randn(n_samples, 2)
    features = np.concatenate([x0, x1, x2, x3, x4, x5, x6, x_noise], axis=-1)
    
    rules = [np.logical_and.reduce([(x0+0.5*x1+x2)**2/(1+np.exp(1+x5+2*x6))<1], axis=0)]
    
    for i in range(len(rules)):
        print('Rule {}: {:.2f}%'.format(i, np.sum(rules[i])/n_samples*100))
    
    labels = np.logical_or.reduce(rules, axis=0)[:,0]

    if mislabel is not None:
        one_array = labels[labels==1]
        mutated_one_array = np.where(np.random.random(one_array.shape) < mislabel, False, one_array)
        labels[labels==1] = mutated_one_array
    
    # Add random noise
    features += np.random.randn(features.shape[0], features.shape[1])*0.01
    print('Positive samples: {:.2f}%'.format(np.sum(labels)/features.shape[0]*100))
    
    rule_data =  None
    
    categorical_info = np.zeros([features.shape[1]]) 
    
    dataset = {
        'variables': features.astype(np.float32),
        'response': labels.astype(np.int8),
        'num_classes': 2,
        'category_info': categorical_info,
        'rule_data': rule_data,
        'feature_names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
        }
    
    np.random.seed(None)
    return dataset
