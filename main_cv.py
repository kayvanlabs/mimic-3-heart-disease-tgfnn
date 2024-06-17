# -*- coding: utf-8 -*-
"""
Performing nested cross-validation on the dataset with machine learning techniques.
The inner cross-validation is used for hyper-parameter tuning.
The outer cross-validation is used to better evaluate the performance of the machine learning techniques.
"""
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
import os
import pickle
import load_dataset
import utils
from generalized_fuzzy_net import GeneralizedFuzzyClassifier
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Run nested cross-validation for hyperparameter tuning on TGFNN and other models.')

    parser.add_argument('-i', '--input', type=str, help='The input file/dir of the dataset.', required=True)
    parser.add_argument('-o', '--output', type=str, help='The output folder.', default='./cv_results')
    parser.add_argument('-n', '--name', type=str, help='The name of the experiment.', default='TGFNN_CV')
    parser.add_argument('--n_folds', type=int, help='The number of folds in the outer cross-validation. Set to more than 1 for evaluating the stability of the rules.', default=1)
    parser.add_argument('--search_iters', type=int, help='The number of search interations in the hyper-parameter tuning.', default=100)
    parser.add_argument('--n_folds_hyper_tuning', type=int, help='The number of folds in the inner cross-validation. This is how many times you split your data, the number depends on the size of your dataset', default=10)
    parser.add_argument('-m', '--max_steps', type=int, help='The maximal steps in training TGFNN', default=20000)
    parser.add_argument('-r', '--random_state', type=int, help='The random state.', default=0)
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode. Set to True if only want to debug the code.')
    parser.add_argument('-p', '--pickled', action='store_true', help='If the input file is a pickle file.')

    return parser.parse_args()

def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)


def nested_cross_validation(X, y, num_classes, category_info, feature_names, rule_data, model_name, 
                            n_folds, n_folds_hyper_tuning, search_iters, max_steps, split_method, 
                            random_state):
    """
    Nested cross-validation.

    Parameters
    ---------- 
    X : np.ndarray. A np.array of features with a shape (the number of samples, the number of features)
    y : np.dnarray. A np.ndarray of labels with a shape (the number of samples,)
    num_classes : Int. The number of classes
    category_info : A np.array with a shape (the number of features).  entry category_info[i] = 0 means the i-th variable 
        is a continous variable. entry category_info[i] > 0 means the i-th variable is a categorical variable, and 
        the value is the number of levels. This information is only used in the proposed machine learning technique.
    feature_names : A list of feature names.
    rule_data : A list of existing rules
    model_name : Str. A string of model name. Options: 'RF', 'XGB', 'SVM', 'EBM', 'GFN_cv', 'DT', 'LR', 'NB', 'FLN', 'GFN'
    n_folds : Int. The number of folds in the outer cross-validation.
    n_folds_hyper_tuning : Int. The number of folds in the inner cross-validation.
    search_iters : Int. The number of search interations in the hyper-parameter tuning.
    max_steps : Int. The maximal steps in optimizing GFN algorithm.
    split_method : Str. It indicates how the train/val/test data split should be performed.
        Options include 'patient_wise', and 'sample_wise'. 'sample_wise' is the regular split. 
        For 'patient_wise', data samples from the same patient should be put into the same data set.
    random_state : Int. Random state.
    
    Returns
    -------
    fold_classifiers : A list of trained classifier objects from the outer cross-validation.
    eval_series : A pd.Series contains evaluation metrics from the outer cross-validation.
    roc_values : A dictionary with keys: 'fpr_test', 'tpr_test', 'auc_test' from the outer cross-validation. 
        It can help draw the ROC curve and its confidence interval from the outer cross-validation.

    """
    fold_classifiers = []
    
    ss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.20, random_state=random_state) # Split the dataset for the outer cross-validation
    cv_network = ['RF', 'XGB', 'SVM', 'EBM', 'GFN', 'DT'] # models that require hyperparameter tuning
    

    for index in range(n_folds):

        # I do not think that this split is necessary
        X_train, y_train, X_val, y_val = utils.split_dataset(ss, X, y, split_method, index=index)

        # compute class weights on train set
        classes, counts = np.unique(y_train, return_counts=True)
        weights = counts.sum() / (2 * counts)
        weights_dict = dict(zip(classes, weights))


        # Standardize input data for non-GFN models 
        if not model_name.startswith('GFN'):
            scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
        
        # Instantiate model
        if model_name == 'SVM':
            base = SVC(probability=True, random_state=random_state)
            grid = utils.create_SVM_grid()

        elif model_name == 'RF': # random forest
            base = RandomForestClassifier(
                class_weight=weights_dict,
                random_state=random_state
                )
            grid = utils.create_RF_grid()

        elif model_name == 'XGB': # XGBoost
            import xgboost
            base = xgboost.XGBClassifier()
            grid = utils.create_XGBoost_grid()

        elif model_name == 'LR': # Logistic regression
            classifier = LogisticRegression(
                solver='lbfgs',
                class_weight=weights_dict,
                random_state=random_state,
                max_iter=5000
                )
            
        elif model_name == 'NB': # Naive bayes
            classifier = GaussianNB()

        elif model_name == 'EBM': # Explainable boosting machine
            from interpret.glassbox import ExplainableBoostingClassifier
            base = ExplainableBoostingClassifier(
                random_state=random_state
                )
            grid = utils.create_EBM_grid(num_classes==2)

        elif model_name == 'DT':
            base = sklearn.tree.DecisionTreeClassifier()
            grid = utils.create_DT_grid()

        elif model_name == 'GFN': # Generalized fuzzy network
            base = GeneralizedFuzzyClassifier(
             n_classes=num_classes,
             max_steps=max_steps, #100000
             category_info=category_info,
            #  batch_size = 1,
             report_freq = 50, # 50
             patience_step = 1000, # 500
             random_state=random_state,
             epsilon_training=True,
             binary_pos_only=True,
             weighted_loss=weights,
             split_method=split_method,
             verbose=0,
             init_rule_index = rule_data,
             rule_data = rule_data
             ) 
            grid = utils.create_GFN_grid()


        # Hyper-parameter tuning
        if model_name in cv_network:
            import platform
            if platform.system() == 'Windows':
                n_jobs = 1
            else:
                n_jobs = -1
            
            # Inner cross-validation !!! consider using HalvingGridSearchCV for faster search
            classifier = RandomizedSearchCV(estimator=base, param_distributions=grid, 
                                           n_iter=search_iters, cv=n_folds_hyper_tuning, 
                                           verbose=1, random_state=random_state,
                                           n_jobs=n_jobs, scoring=sklearn.metrics.make_scorer(multiclass_roc_auc_score)) # adjust scoring metric here
  
        # Fit the random search model
        classifier.fit(X_train, y_train)
        
        if model_name in cv_network: 
            best_classifier = classifier.best_estimator_
            
            # save cv results
            cv_results = pd.DataFrame(classifier.cv_results_)
            cv_results.to_csv(os.path.join(save_path, f'cv_results_{model_name}.csv'))
            
        else:
            best_classifier = classifier
        
        fold_classifiers.append(best_classifier) # Save best classifier from each outer fold

    return fold_classifiers
    

def fit_and_eval(model, i, X_train, y_train, X_test, y_test, num_classes, model_name):
    model.fit(X_train, y_train)
    params = str(model.get_params())

    # evaluate
    _, train_metrics, _, _, _, fpr_train, tpr_train, train_conf_mat = utils.cal_acc(model, X_train, y_train, num_classes>2)
    _, test_metrics, _, _, _, fpr_test, tpr_test, test_conf_mat = utils.cal_acc(model, X_test, y_test, num_classes>2)

    conf_mat = [train_conf_mat, test_conf_mat]
    
    # compile metrics
    roc_values = {'fpr_test': fpr_test, 'tpr_test': tpr_test, 'auc_test': test_metrics[5],
                    'fpr_train': fpr_train, 'tpr_train': tpr_train, 'auc_train': train_metrics[5]}

    
    metrics = pd.DataFrame([train_metrics, test_metrics], 
                            columns=['Accuracy', 'Recall', 'Specificity', 'Precision', 'NPV', 'F1', 'AUC', 'AUCPR'], 
                            index=['Train', 'Test'])
    metrics = metrics.reset_index(names='Set')
    metrics['Model'] = model_name
    metrics['model_index'] = i
    metrics['Parameters'] = params

    return model, metrics, roc_values, conf_mat


def extract(dataset):
    '''
    Extract the variables from the dataset dictionary.
    '''

    split_method = dataset['split_method'] if 'split_method' in dataset else 'sample_wise'

    if 'X_train' in dataset.keys():
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        
    else:
        data = np.array(dataset['variables'])
        labels = np.array(dataset['response'])
        ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
        X_train, y_train, X_test, y_test = utils.split_dataset(ss_train_test, data, labels, split_method, index=0)

    category_info = dataset['category_info']
    num_classes = dataset['num_classes']
    rule_data = dataset.get('rule_data') 
    feature_names = dataset.get('feature_names')

    return X_train, y_train, X_test, y_test, num_classes, category_info, feature_names, rule_data, split_method


if __name__ == '__main__':

    ################################################################################################################
    # Experiment confgurations
    ################################################################################################################

    args = parse_args()

    # Create output folders
    save_path = os.path.join(args.output, args.name)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Parameters
    n_folds = args.n_folds
    search_iters = args.search_iters
    n_folds_hyper_tuning = args.n_folds_hyper_tuning
    random_state = args.random_state
    model_set = ['GFN', 'LR', 'RF']

    if args.debug:
        n_folds = 1
        search_iters = 1
        n_folds_hyper_tuning = 2
        max_steps = 100
    else:
        n_folds = n_folds
        search_iters = search_iters
        n_folds_hyper_tuning = n_folds_hyper_tuning
        max_steps = args.max_steps

    # Print experiment configurations
    print('######################################')
    print('Experiment Name:', args.name)
    print('n_folds:', n_folds)
    print('search_iters:', search_iters)
    print('n_folds_hyper_tuning:', n_folds_hyper_tuning)
    print('max_steps:', max_steps)
    print('######################################')


    # Load data into dataset dictionary
    if args.pickled:
        dataset = pickle.load(open(args.input, 'rb'))
    else:
        dataset = load_dataset.load_data(args.input)
        pickle.dump(dataset, open(os.path.join(save_path, 'dataset.pkl'), 'wb'))
        print(f'Dataset saved to {os.path.join(save_path, "dataset.pkl")}')


    ################################################################################################################
    # Run CV hyperparameter tuning and evaluation for each model type
    ################################################################################################################

    all_model_metrics = pd.DataFrame()

    for model_name in model_set:

        print(model_name)

        # Extract the variables and parameters from the dataset dictionary
        X_train, y_train, X_test, y_test, num_classes, category_info, feature_names, rule_data, split_method = extract(dataset)

        # Identify optimal hyperparameters via nested cross-validation on the training set
        fold_classifiers = nested_cross_validation(
            X_train, y_train, num_classes, category_info, feature_names, 
            rule_data, model_name, n_folds, n_folds_hyper_tuning, 
            search_iters, max_steps, split_method, random_state)


        # Retrain all the models on the whole training set and evaluate
        fitted_fold_classifiers = []
        all_metrics = []
        all_roc_values = {'fpr_test': [], 'tpr_test': [], 'auc_test': [],
                        'fpr_train': [], 'tpr_train': [], 'auc_train': []}
        all_conf_mat = []
        
        # For every hyperparameter-tuned model (n_folds of them)...
        for i in range(len(fold_classifiers)):

            model = fold_classifiers[i]
            fitted_model, metrics, roc_values, conf_mat = fit_and_eval(model, i, X_train, y_train, X_test, y_test, num_classes, model_name)
            all_metrics.append(metrics)
            fitted_fold_classifiers.append(fitted_model)
            all_conf_mat.append(conf_mat)

            # compile roc values for plotting
            for key in all_roc_values.keys():
                all_roc_values[key].append(roc_values[key])


        # Save the models
        pickle.dump(fitted_fold_classifiers, open(os.path.join(save_path, f'{len(fitted_fold_classifiers)}_trained_{model_name}.mdl'), 'wb'))

        # Collate performance metrics
        metrics = pd.concat(all_metrics, axis=0, ignore_index=True)
        all_model_metrics = pd.concat([all_model_metrics, metrics], axis=0, ignore_index=True)

        # Print best model metrics (based on AUC)
        idx = metrics[metrics['Set'] == 'Test']['AUC'].idxmax() # you can select the best model based on other metrics
        i = metrics.iloc[idx,:]['model_index']
        print(f'BEST {model_name} MODEL PERFORMANCE')
        print(metrics[metrics['model_index'] == i][['model_index', 'Accuracy', 'Recall', 'Specificity', 'Precision', 'NPV', 'F1', 'AUC', 'AUCPR']])
        print('Train and test confusion matrices:')
        print(all_conf_mat[i][0])
        print(all_conf_mat[i][1])

        # Draw the ROC curve and the confidence intervals (if multiple outer folds requested).
        utils.draw_ROC_curves(all_roc_values['tpr_train'], all_roc_values['fpr_train'], all_roc_values['auc_train'],
                          model_name, os.path.join(save_path, f'ROC_train_{model_name}.png'))
        utils.draw_ROC_curves(all_roc_values['tpr_test'], all_roc_values['fpr_test'], all_roc_values['auc_test'],
                          model_name, os.path.join(save_path, f'ROC_test_{model_name}.png'))
        

    # Save the metrics
    metrics_file_name = os.path.join(save_path, f'results.csv')
    all_model_metrics.to_csv(metrics_file_name, index=False)
    print(f'All model performance metrics saved to {metrics_file_name}')