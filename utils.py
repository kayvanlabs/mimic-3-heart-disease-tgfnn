# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:38:31 2020

@author: hemingy
"""
import numpy as np
import sklearn
from scipy.stats import loguniform
from itertools import repeat
import matplotlib.pyplot as plt


def draw_ROC_curves(tprs_list, fprs_list, aucs_list, model_name, output_name):
    """ Rraw ROC curves with confidence interval.

    """
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs_list = []
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(len(aucs_list)):
        interp_tpr = np.interp(mean_fpr, fprs_list[i], tprs_list[i])
        interp_tpr[0] = 0.0
        interp_tprs_list.append(interp_tpr)
   
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
   
    mean_tpr = np.mean(interp_tprs_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_list)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label='Mean ROC\n' + r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
   
    std_tpr = np.std(interp_tprs_list, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
   
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f"ROC curve for {model_name}")
    ax.legend(loc="lower right")
    plt.savefig(output_name, dpi=300)


def show_metrics(metrics, show_value_list=None):
    """ Calculate the average and standard deviation from multiple repetitions and format them.
    """
    eval_m, eval_s = np.nanmean(metrics, 0), np.nanstd(metrics,0)
    show_results(eval_m, s=eval_s, show_full=False)  
    for i in range(eval_m.shape[0]):
        show_value_list.append('{:.3f} ({:.3f})'.format(eval_m[i], eval_s[i]))
    return show_value_list

   
def split_dataset(stratified_split, data, labels, split_method, index=0):        
    """ Split the dataset into the training set and test set.

    Parameters:
    ---------- 
    stratified_split : An instance of the object sklearn.model_selection.StratifiedShuffleSplit.
    data : np.ndarray. A np.array of features with a shape (the number of samples, the number of features).
    labels : np.dnarray. A np.ndarray of labels with a shape (the number of samples,).
    split_method : Str. It indicates how the train/val/test data split should be performed.
        Options include 'patient_wise', and 'sample_wise'. 'sample_wise' is the regular split. 
        For 'patient_wise', data samples from the same patient should be put into the same data set.
    index : Int. The index of the repetition.

    Returns:
    ---------- 
    X_train : np.ndarray with a shape (the number of training samples, the number of features). Training features.
    y_train : np.ndarray with a shape (the number of training samples,). Training labels.
    X_test : np.ndarray with a shape (the number of training samples, the number of features). Testing features.
    y_test : np.ndarray with a shape (the number of training samples,). Testing features.
    
    """
    if split_method == 'patient_wise':
        uids = data[:, 0]        
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==0]).difference(uids_HT_VAD_set)
        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.zeros(uids_too_well_arr.shape[0])], axis=0)
    
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)
    
        train_index = [i for i in range(data.shape[0]) if data[i,0] in uids_train]
        test_index = [i for i in range(data.shape[0]) if data[i,0] in uids_test]
        
    elif split_method == 'sample_wise':
        # Split the data for nested cross-validation on the training set
        index = list(stratified_split.split(data, labels))[index]
        train_index = index[0]
        test_index = index[1]
    
    else:
        raise NotImplementedError
        
    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0)  
        
    return X_train.astype(np.float32), y_train.astype(np.int32), X_test.astype(np.float32), y_test.astype(np.int32)

     
def standardize(features, scaler, category_list):
    if scaler is not None:
        if category_list is not None:
            features_continous = features[:, category_list==0]
            scaled_features = features.copy()
            scaled_features[:, category_list==0] = scaler.transform(features_continous)
        else:
            scaled_features = scaler.transform(features)
    else:
        scaled_features = features
        
    return scaled_features


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data
            
            
def show_results(m, s=None, show_full=False, name=None):
    """ Present the performance of the model.

    Parameters
    ----------
    m : np.ndarray. Mean value of the evaluation metrics including accuracy, sensitivity, specificity, precision, f1-score, auc, and aucpr.
    s : np.ndarry, optional. Standard deviation of the evaluation metrics including accuracy, sensitivity, specificity, precision, f1-score,
        auc, and aucpr.
    show_full : boolean, optional. Whether to show the full evaluation metrics The default is False.
    name: string.
    -------
    None.

    """
    # Show the percentage
    m = m*100
    if s is not None:
        s = s*100
        
    if name:
        print(f'{name} evaluation metrics:')
    if show_full:
        string = '\t Mean acc is {:.2f}, sen is {:.2f}, spe is {:.2f}, prec is {:.2f}, f1 is {:.2f}, auc is {:.2f}, aucpr is {:.2f}'.\
               format(m[0],m[1],m[2],m[3],m[4],m[5],m[6])
        print(string)
        
        if s is not None:            
            string = '\t Std acc is {:.2f}, sen is {:.2f}, spe is {:.2f}, prec is {:.2f}, f1 is {:.2f}, auc is {:.2f}, aucpr is {:.2f}'.\
                   format(s[0],s[1],s[2],s[3],s[4],s[5],s[6])
            print(string)
    else:
        if s is None:
            string = '\t auc is {:.2f}, aucpr is {:.2f}, F1-score is {:.2f}'.\
                   format(m[5],m[6],m[4])
        else:
            string = '\t auc is {:.2f} ({:.2f}), aucpr is {:.2f} ({:.2f}), F1-score is {:.2f} ({:.2f})'.\
                   format(m[5],s[5],m[6],s[6],m[4],s[4])
        print(string)
            
            
def indices_to_one_hot(data, n_classes=None):
    """ Convert an iterable of indices to one-hot encoded labels.
    

    Parameters
    ----------
    data : list. A list of integers.
    n_classes: int. The number of classes.

    Returns
    -------
    One-hot encoded labels.
    
    """
    if n_classes is None:
        n_classes = np.max(data) + 1
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def calibration_curve(model, features, labels, n_bins=10):
    """ Model prediction and evaluation metrics calculation (For classification only).
    
    Parameters
    ----------
    model : An estimator object. The trained model.
    features : np.ndarray. Normalized features.
    labels :  np.ndarray. Labels.
    multiple : boolean. Whether it is a multi-class classification task.
    """
    from sklearn.calibration import calibration_curve

    if not isinstance(model, list):
        #predictions = model.predict(features)
        probs = model.predict_proba(features)
        predictions = np.argmax(probs, axis=-1)
        
    else:
        # A list of classifiers
        probs_list = []
        for voter in model:
            probs = voter.predict_proba(features)
            probs_list.append(probs)
        probs = np.mean(np.stack(probs_list, axis=-1), axis=-1)
        predictions = np.argmax(probs, axis=-1)

    prob_true, prob_pred = calibration_curve(labels, predictions, n_bins=n_bins)

    return prob_true, prob_pred


def cal_acc(model, features, labels, multiple):
    """ Model prediction and evaluation metrics calculation (For classification only).
    
    Parameters
    ----------
    model : An estimator object. The trained model.
    features : np.ndarray. Normalized features.
    labels :  np.ndarray. Labels.
    multiple : boolean. Whether it is a multi-class classification task.

    Returns
    -------
    predictions : np.ndarray. Predictions from the traine model on the given features.
    metrics: A tuple of float. They are recall, specificity, precision, accuracy, f1, auc, aucpr, respectively.
        For multi-class classification, the values from individual classes are averaged.

    """
    if not isinstance(model, list):
        #predictions = model.predict(features)
        probs = model.predict_proba(features)
        predictions = np.argmax(probs, axis=-1)
        
    else:
        # A list of classifiers
        probs_list = []
        for voter in model:
            probs = voter.predict_proba(features)
            probs_list.append(probs)
        probs = np.mean(np.stack(probs_list, axis=-1), axis=-1)
        predictions = np.argmax(probs, axis=-1)
    
    # Calculate matrix
    if multiple:
        average_method = 'macro'
    else:
        average_method = 'binary'

    recall = sklearn.metrics.recall_score(labels, predictions, average=average_method) # also called sensitivity
    precision = sklearn.metrics.precision_score(labels, predictions, average=average_method, zero_division=0) # also called positive pretictive value
    npv = sklearn.metrics.precision_score(1-labels, 1-predictions, average=average_method, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-labels, 1-predictions, average=average_method)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    f1 = sklearn.metrics.f1_score(labels, predictions, average=average_method)
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    
    if multiple:
        aucpr = sklearn.metrics.average_precision_score(indices_to_one_hot(labels), 
                                                        probs, average=average_method)
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels, np.max(labels)+1), probs,
                                            average=average_method, multi_class='ovr')
        fpr, tpr = None
    else:
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels), probs)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:,1])

    metrics = np.array([accuracy, recall, specificity, precision, npv, f1, auc, aucpr])
    metrics_name = ['accuracy',  'recall', 'specificity', 'precision', 'npv',
                    'f1', 'auc', 'aucpr']
    return predictions, metrics, metrics_name, probs, labels, fpr, tpr, confusion_matrix


def create_RF_grid():
    """ Search grid for hyper-parameter tuning on random forest model.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Create the grid
    param_grid = {'n_estimators': [int(x) for x in np.linspace(100, 1000, 5)],
                # Maximum number of levels in tree
                'max_depth': [int(x) for x in np.linspace(3, 10, 8)],
                # Number of features to consider at every split
                'min_samples_split': [int(x) for x in np.linspace(2, 200, 9)],
                # Minimum decrease in impurity required for split to happen
                'max_features': ['sqrt', 'log2'],
                'min_samples_leaf': [int(x) for x in np.linspace(1, 200, 9)]
                }
    return param_grid


def create_DT_grid():
    """ Search grid for hyper-parameter tuning on random forest model.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Create the grid
    param_grid = {
                'max_depth': [int(x) for x in np.linspace(5, 25, 10)] + [None],
                # Minimum number of samples required to split a node
                'min_samples_split': [int(x) for x in np.linspace(2, 10, 9)]
                # Minimum decrease in impurity required for split to happen
                }
    return param_grid


def create_EBM_grid(binary):
    """ Search grid for hyper-parameter tuning on Explainable Boosting Classifier.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Create the grid
    if binary:
        param_grid = {
            'max_bins': [256, 512, 1024, 2048],
            'max_interaction_bins': [8, 16, 32, 64],
            # 'interactions': [0.5, 0.75, 0.95],
            'validation_size': [0.1, 0.2, 0.3],
            'outer_bags': [5, 10, 14, 18],
            'inner_bags': [0, 5, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_rounds': [1000, 5000, 10000],
            'early_stopping_rounds': [25, 50, 100, 150],
            'early_stopping_tolerance': [0.0, 0.0001, 0.001, 0.01],
            'min_samples_leaf': [2, 4, 8],
            'max_leaves': [3, 5, 7]
        }
    else:
        # Multiclass with interactions currently not supported in interpret.glassbox.
        param_grid = { # Method to bin values for pre-processing.
                   # Learning rate for boosting
                   'learning_rate': loguniform(1e-3, 1),
                    } 
    return param_grid


def create_SVM_grid():
    """ Search grid for hyper-parameter tuning on SVM with non-linear kernel.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  # Regularization parameter.
                  'C': loguniform(1e-1, 1e3), 
                  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                  'gamma': loguniform(1e-4, 1e0)
                  }
    return param_grid


def create_XGBoost_grid():
    """ Search grid for hyper-parameter tuning on XGBoost classifier.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    param_grid = {'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
                # Maximum tree depth for base learners
                'max_depth': [int(x) for x in np.linspace(3, 10, 8)],
                # Learning rate
                'eta': [x for x in np.linspace(0.1, 1, 10)],
               }
    return param_grid


def create_GFN_grid():
    """ Search grid for hyper-parameter tuning on generalized fuzzy network.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    param_grid = {
        'min_epsilon1': [x for x in np.linspace(0.01,0.99,30)],
        'min_epsilon2': [x for x in np.linspace(0.01,0.99,30)],
        'min_epsilon3': [x for x in np.linspace(0.01,0.99,30)],
        'n_rules': [30,40],
        # 'learning_rate': loguniform.rvs(1e-4, 1e-1, size=30),
        'sparse_regu': loguniform.rvs(1e-3, 1e1, size=30),
        'corr_regu': loguniform.rvs(1e-4, 1e1, size=30),
        # 'infer_regu': loguniform.rvs(1e-4, 1e1, size=30),
        'contradict_regu': loguniform.rvs(1e-3, 1e1, size=30),
        # 'batch_size': [10, 25, 50, 100]
    }
    
    return param_grid