# -*- coding: utf-8 -*-
"""
Rule visualization and encoding function visualization for a single model.

Note: May only work when binary_pos_only is True.

Created on Mon Nov 25 14:55:32 2019

@author: hemingy
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import load_dataset
import sys
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Rule visualization and encoding function visualization for a single model.')
    parser.add_argument('-m', '--model', type=str, help='The path to the model to be visualized.', required=True)
    parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset used to train and evaluate the model. Can either be directory of files or pickled file.', required=True)
    parser.add_argument('-n', '--name', type=str, help='The name of the experiment/model to be visualized.', required=True)
    parser.add_argument('-p', '--pickled', action='store_true', default=True, help='Whether the dataset is pickled.')
    args = parser.parse_args()
    return args
            
            
def draw_rules_from_classifier(classifier, params, save_prefix,
                               keep_irrelevant_variable=False, 
                               filter_redundant_rules=True,
                               filter_similarity_threshold=0.9,
                               show_encoding=False):
    """
    Rule visualization.

    Parameters
    ----------
    classifier : An instance of GeneralizedFuzzyClassifier class.
    params : An instance of Param object, which stores hyper-parameters of the network and information 
        about the dataset. The keys in the Param object includes:
        (hyper-parameters of the network): binary_pos_only, n_rules, n_classes
        (information about the dataset): feature_names, category_info
    keep_irrelevant_variable : Boolean, optional. Whether to keep the variables with little contributions. 
        The default is False.
    filter_redundant_rules : Boolean, optional. Whether to filter out redundant rules based on correlation between 
        pairs of rules. The default is True.
    filter_similarity_threshold : Float, optional. If filter_redundant_rules is True, if two rules have a 
        correlation higher than the threshold, only one will be kept. The default is 0.9.
    save_prefix : Str. Prefix of the generated outputs to be saved.
    show_encoding : Boolean, optional. Whether to draw the trained membership functions of individual continuous
        variables. The default is False.

    Returns
    -------
    None.

    """
    
    rules, rules_all, attention_mask, connection_mask, encoding_value_details = extract_rules_from_net(
        classifier.estimator, params, classifier.scaler, 
        keep_irrelevant_variable=keep_irrelevant_variable,
        filter_similarity_threshold=filter_similarity_threshold)
    
    # Encoding Visualization
    rule_var_name = rules.index
    rule_var_name = set([name.split('_low')[0] for name in rule_var_name if 'low' in name])
    
    if show_encoding:   
        continous_variable_name = [params.feature_names[index] \
                                   for index in range(len(params.category_info)) \
                                    if params.category_info[index]==0]
        
        encoding = pd.DataFrame(np.transpose(encoding_value_details), index=continous_variable_name,
                                columns=['low', 'medium_left', 'medium_right', 'high'])
        pd.options.display.float_format = '{:.2f}'.format
        print(encoding)
        
        # Draw membership functions for individual continuous variables
        for index, var_name in enumerate(continous_variable_name):
            if var_name in rule_var_name:
                draw_membership_function(encoding.iloc[index, 0], encoding.iloc[index, 1], 
                                         encoding.iloc[index, 2], encoding.iloc[index, 3], 
                                         output_path=save_prefix, n_points=1000, 
                                         epsilon=params.epsilon, variable_name=var_name)

    rule_visual_name = save_prefix + '_rules_visualization.png'
    if filter_redundant_rules:
        rule_visualization(rules, name=rule_visual_name)
    else:
        rule_visualization(rules_all, name=rule_visual_name)
    plt.close()
                

def extract_rules_from_net(net, params, scaler, 
                           keep_irrelevant_variable=False,
                           filter_similarity_threshold=0.9):   
    """ Extract interpretable rules from the trained network

    Parameters
    ----------
    params : Params object. 
    model_path : Str. Path to the saved model.
    train_features : np.ndarray. Training features used to train the model. It is used to 
        calculate the mean and variance we used for data pre-processing.
    keep_irrelevant_variable : Boolean. Whether to keep irrelevant variables in rule visualization.

    Returns
    -------
    None.

    """
    row_names_continous = []
    row_names_categorical = []
    for i in range(len(params.category_info)):
        if params.feature_names is None:
            if params.category_info[i] == 0:
                row_names_continous += [f'x{i}_low', f'x{i}_medium', f'x{i}_high']
            else:
                row_names_categorical += [f'x{i}_level{j}' for j in range(params.category_info[i])]
        else:
            feature_name = params.feature_names[i]
            if params.category_info[i] == 0:
                row_names_continous += [f'{feature_name}_low', f'{feature_name}_medium', f'{feature_name}_high']
            else:
                row_names_categorical += [f'{feature_name}_level{j}' for j in range(params.category_info[i])]
    row_names = row_names_continous + row_names_categorical
    
    ## For continous variables
    # Extract the way that varaibles are encoded
    encoding_values, extract_encoding_details = extract_encoding_intervals(net, scaler)
    encoding_column_continous = np.expand_dims(encoding_values.flatten('F'), axis=1)
    encoding_column_categorical = []
    category_levels = params.category_info[params.category_info>0]
    for n_levels in category_levels:
        encoding_column_categorical += [i for i in range(n_levels)]
    encoding_column_categorical = np.expand_dims(np.array(encoding_column_categorical), axis=-1)
    encoding_column = np.concatenate([encoding_column_continous,
                                      encoding_column_categorical], axis=0)

    # Extract the attention mask and connection mask
    # the entry in relation is calculated by multiplying the corresponding entry in the attention mask with
    # the corresponding entry in the connection mask     
    attention_mask, connection_mask, relation_mat = extract_relations(net, params)

    # Extract how the rules contribute to the prediction
    if params.binary_pos_only:
        out_layer = net.layer3.weight.detach().numpy()**2
        out_layer = out_layer.reshape(-1, 1)
        row_names.extend(['pos_class_contribution'])
    else:
        out_layer = net.layer3.weight.detach().numpy()**2
        row_names.extend([f'{i}' for i in range(params.n_classes)])
        # out_layer[:params.n_rules] *= -1
    
    
    weighted_out_layer = out_layer/np.max(np.abs(out_layer)) # This is important to note
    out_row = np.insert(weighted_out_layer, 0, np.nan, axis=0)

    rules = np.concatenate([encoding_column, relation_mat], axis=-1)
    rules = np.concatenate([rules, out_row.T], axis=0)
    
    # Remove redundant rules based on correlation on pairs of rules. 
    merged_rules, keep_index = remove_redundant_rules(rules, params,
                                                      information_threshold=0.1,
                                                      similarity_threshold=filter_similarity_threshold)
        
    # Build the row names and column names
    if params.binary_pos_only:
        all_column_names = ['encoding']
        for i in range(params.n_rules):
            all_column_names.append(f'Rule_{i}')
    else:
        all_column_names = ['encoding']
        for i in range(params.n_rules):
            all_column_names.append(f'Rule_{i}')
        
    column_names = ['encoding']
    for i, index in enumerate(keep_index):
        column_names.append(f'Rule_{i+1}')
        
    # Table with all rules extracted from the network
    rules_all =  pd.DataFrame(rules, columns=all_column_names, index=row_names)
    rules_all = rules_all.round(3)
    
    # After redundant or non-informative rules are filtered, we want to remove variables
    # that have little contribution to any rules. Those variables are called irrelevant variables 
    # because they have little predictive value to the target classification task.
    if not keep_irrelevant_variable and merged_rules.shape[1]>1:
        rules = filter_irrelevant_variables(merged_rules, row_names, column_names, params.category_info,
                                                        filter_threshold=0.5)
    else:
        rules =  pd.DataFrame(merged_rules, columns=column_names, index=row_names)
    rules = rules.round(3)

    return rules, rules_all, attention_mask, connection_mask, extract_encoding_details


def f(value, epsilon):
    # Re-formulated membership function in this proposed algorithm
    # find indices where value >0 and <0
    i1 = value>=0
    i2 = value<0
    out = np.zeros(value.shape)
    out[i1] = value[i1] + epsilon*np.log(np.exp(-value[i1]/epsilon)+1)
    out[i2] = epsilon*np.log(1+np.exp(value[i2]/epsilon))
    return out
    

def draw_membership_function(a1, a2, b1, b2, output_path, variable_name='x', n_points=100, epsilon=0):
    # Build membership function
    start = a1-2*(b1-a2)
    end = b2+2*(b1-a2)
    step = (end-start)/n_points
    x = start + np.array(range(n_points))*step 
    
    y_low = f((a2-x)/(a2-a1), epsilon) - f((a1-x)/(a2-a1), epsilon)
    y_medium = f((x-a1)/(a2-a1), epsilon) - f((x-a2)/(a2-a1), epsilon) + f((b2-x)/(b2-b1), epsilon) - f((b1-x)/(b2-b1), epsilon) - 1
    y_high = f((x-b1)/(b2-b1), epsilon) - f((x-b2)/(b2-b1), epsilon)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y_low, label='Low')
    plt.plot(x, y_medium, label='Medium')
    plt.plot(x, y_high, label='High')
    
    plt.legend(fontsize=16)
    plt.ylabel('Membership Value', fontsize=18)
    plt.xlabel(variable_name, fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.savefig(output_path + f'_{variable_name}_{epsilon:.2f}.png', dpi=300, bbox_inches="tight")  


def extract_encoding_intervals(net, scaler):
    """ Extract the threshold for variable encoding.

    Parameters
    ----------
    net : Net object. Trained network.
    train_features : np.ndarray. Training features used to train the model. It is used to 
        calculate the mean and variance we used for data pre-processing.

    Returns
    -------
    encoding_values: np.ndarray. Each row givens the 'fuzzy' threshold for low level, medium level, and high level.

    """
    a2_list = net.layer1.weight[0,:] - net.layer1.weight[1,:].pow(2)
    b1_list = net.layer1.weight[0,:] + net.layer1.weight[1,:].pow(2)
    a1_list = a2_list - 0.01 - net.layer1.weight[2,:].pow(2)
    b2_list = b1_list + 0.01 + net.layer1.weight[3,:].pow(2)
        
    low_value = scaler.inverse_transform(a1_list.detach().numpy().reshape([1, -1]))
    medium_left = scaler.inverse_transform(a2_list.detach().numpy().reshape([1, -1]))
    medium_right = scaler.inverse_transform(b1_list.detach().numpy().reshape([1, -1]))
    high_value = scaler.inverse_transform(b2_list.detach().numpy().reshape([1, -1]))
    
    medium_value = (medium_left + medium_right)*0.5
    encoding_values_for_visual = np.concatenate([low_value, medium_value, high_value], axis=0)
    encoding_values = np.concatenate([low_value, medium_left, medium_right, high_value], axis=0)
    
    return encoding_values_for_visual, encoding_values


def extract_relations(net, params):
    """ Extraction relation matrix.
    
    Each element indicates the contribution of one concept.
    
    Parameters
    ----------
    net : Net object. Trained network.
    params : Params object. 
    
    Returns
    -------
    attention_mask: np.ndarray. Each eatry gives the contribution of one concept to the corresponding variable in one rule.
    connection_mask: np.ndarray. Each eatry gives the contribution of one input variable to one rule. 
    relation_mat: np.ndarray. Each element indicates the contribution of one concept to one rule
    
    """
    # Extract the trained weights
    attention_continous = net.layer2.attention_continuous.detach().numpy()
    attention_categorical = net.layer2.attention_categorical.detach().numpy()
    connection = net.layer2.connection.detach().numpy()
    
    # Convert the weights to masks by applying the tanh function
    if params.binary_pos_only:
        attention_mask_continous = np.reshape((np.tanh(attention_continous)+1)/2, [-1, params.n_rules])
        attention_mask_categorical = np.reshape((np.tanh(attention_categorical)+1)/2, [-1, params.n_rules])
    else:
        attention_mask_continous = np.reshape((np.tanh(attention_continous)+1)/2, [-1, params.n_rules])
        attention_mask_categorical = np.reshape((np.tanh(attention_categorical)+1)/2, [-1, params.n_rules])
    connection_mask = (np.tanh(connection)+1)/2
    
    # In the connection matrix, each entry indicates the contribution of one variable.
    # As a result, we need to expand the contribution of one variable to match the number 
    # of concepts from this variable.
    connection_mat_continous = []
    connection_mat_categorical = []
    
    num_continous_x = np.sum(params.category_info==0)
    temp_num_categorical_x = 0
    for i in range(len(params.category_info)):
        if params.category_info[i] == 0:
            connection_mat_continous += [connection_mask[i-temp_num_categorical_x,:]]*3
        else:
            connection_mat_categorical += [connection_mask[num_continous_x+temp_num_categorical_x,:]]*params.category_info[i]
            temp_num_categorical_x += 1
    
    connection_mat_continous = np.stack(connection_mat_continous, axis=0)
    relation_mat_continous = attention_mask_continous*connection_mat_continous
    
    if len(connection_mat_categorical)>0:
        connection_mat_categorical = np.stack(connection_mat_categorical, axis=0)
        relation_mat_categorical = attention_mask_categorical*connection_mat_categorical
        
        attention_mask = np.concatenate([attention_mask_continous, attention_mask_categorical], axis=0)
        relation_mat = np.concatenate([relation_mat_continous, relation_mat_categorical], axis=0)
        return attention_mask, connection_mask, relation_mat
    else:
        return attention_mask_continous, connection_mask, relation_mat_continous


def filter_irrelevant_variables(mat, row_names, columns, category_info, filter_threshold=0.1):
    """ Filter the variables with little conribution to the classification.
    
    The contribution of each variables (including all concepts in all rules) will be summed up.
    Then the variables with contribution smaller than the threshold will be filtered out.
    
    Parameters
    ----------
    mat : np.ndarray. Extracted rules.
    row_names : A list. Name of variables concepts for the mat.
    columns: A list. Column names. 
    params : Params object. 
    filter_threshold : A float. The threshold used to filter the irrelevant varaible.

    Returns
    -------
    filtered_rule_mat: np.ndarray. Extracted rules after irrelevant variables are removed.
    filtered_row_names : A list of strings. Name of variables concepts after irrelevant variables are removed.
    """

    filtered_row_names = []
    filtered_mat = []
    n_continous_variables = np.sum(category_info==0)
    n_variables = len(category_info)
    category_levels = category_info[category_info>0]

    for i in range(n_variables):
        if i<n_continous_variables:
            submat = mat[i*3:(i+1)*3, 1:]
        else:
            # prev = n_continous_variables*3+np.sum(category_levels[:i-n_continous_variables]) 
            # submat = mat[prev:prev+category_levels[i-n_continous_variables]]
            prev = n_continous_variables*3+np.sum(category_levels[:i-n_continous_variables]) 
            submat = mat[prev:prev+category_levels[i-n_continous_variables], 1:]
        
        # min_value = np.min(submat, axis=0)
        # if i < n_continous_variables:
        #     submat -= np.stack([min_value]*submat.shape[0], axis=0)    
        min_value = np.min(submat, axis=0)
        submat -= min_value

        if np.max(submat) > filter_threshold:   
            if i < n_continous_variables:
                filtered_mat.append(np.concatenate([mat[i*3:(i+1)*3, 0:1], submat], axis=1))
                filtered_row_names += row_names[i*3:(i+1)*3]
            else:
                # filtered_mat.append(submat-min_value)
                # filtered_row_names += row_names[prev:prev+category_levels[i-n_continous_variables]]
                filtered_mat.append(np.concatenate([mat[prev:prev+category_levels[i-n_continous_variables], 0:1], submat], axis=1))
                filtered_row_names += row_names[prev:prev+category_levels[i-n_continous_variables]]
            
        else:
            if i<n_continous_variables:
                filtered_variable_name = row_names[i*3].split(' ')[0]
            else:
                prev = n_continous_variables*3+np.sum(category_levels[:i-n_continous_variables])     
                filtered_variable_name = row_names[prev].split(' ')[0]
            print(f'{filtered_variable_name} does not contribute to rules.')
        
    if len(filtered_mat)==0:
        print('All variables are filtered.')
        rules =  pd.DataFrame(mat, columns=columns, index=row_names)
        return rules
    else:
        filtered_rule_mat = np.concatenate(filtered_mat, axis=0)
        filtered_rule_mat = np.concatenate([filtered_rule_mat, mat[-1:,:]], axis=0)
        filtered_row_names.append('directions')
        rules =  pd.DataFrame(filtered_rule_mat, columns=columns, index=filtered_row_names)
        return rules
    
    
def remove_redundant_rules(mat, params, information_threshold=0.01, similarity_threshold=0.9):
    """ Filter redundant rules.
    
    If the information of one rule is very small (sum of the concepts' contribution in this rule),
    the rule will be removed.
    
    If we find a group of rules with similar pattern, the rule with largest weight in the inference layer
    will be selected.
    """
    relation_mat = mat[:-1, 1:]
    rule_weight = mat[-1, 1:]
    # Remove rules with low weight to the output layer
    relation_list = []
    rule_weight_list = []
    for i in range(params.n_rules):
        if rule_weight[i] >= information_threshold:
            relation_list.append(relation_mat[:, i])
            rule_weight_list.append(rule_weight[i])
    relation_mat = np.stack(relation_list, axis=-1)
    weights = np.array(rule_weight_list)
    
    df = pd.DataFrame(relation_mat)
    corr_mat = df.corr()
    
    rule_index_list = list(range(relation_mat.shape[1]))
    keep_rule_index_list = list(range(relation_mat.shape[1]))
    
    rule_groups = []
    while len(rule_index_list)>0:
        index = rule_index_list.pop(0)
        # If the rule value is not a constant
        if np.max(relation_mat[:, index])>information_threshold:
            corr = corr_mat.iloc[index, :]
            merge_list = [index]
            for j in range(index+1, corr.shape[0]):
                if corr[j] > similarity_threshold and j in keep_rule_index_list:
                    keep_rule_index_list.remove(j)
                    rule_index_list.remove(j)
                    merge_list.append(j)
            rule_groups.append(merge_list)
    
    rule_index_list = []
    for merge_list in rule_groups:
        weight_list = np.take(weights, merge_list)
        # Test whether have the same direction. If not, rules in this group should be removed.
        if np.all(weight_list == np.abs(weight_list)) or np.all(weight_list == -np.abs(weight_list)) :
           sel_index = merge_list[np.argmax(np.abs(weight_list))]
           rule_index_list.append(sel_index)
    
    relation_mat = np.concatenate([relation_mat, np.expand_dims(weights, axis=0)], axis=0)
    filtered_rules = np.take(relation_mat, rule_index_list, axis=1)
    encoding = mat[:, 0:1]
    mat = np.concatenate([encoding, filtered_rules], axis=1)
    return mat, rule_index_list
                

def rule_visualization(table, name='test.png'):
    """ Visualize the extracted rules.

    Parameters
    ----------
    table : pd.DataFrame. Extracted rules. The columns names are the rule indexes and the 
        row names are names of variable concepts.
    name : Str, optional. Name of the saved plot. The default is 'test.png'.

    Returns
    -------
    None.

    """
    rules = table.iloc[:-1,1:]
    directions = table.iloc[-1, 1:]
    columns = rules.columns
    new_columns = []
    for i in range(columns.shape[0]):
        new_columns.append('\n'.join([columns[i].replace('_', ' '), str(directions[i])]))
    rules.columns = new_columns
    n_height = 1
    n_weight = 1
    plt.figure(figsize=(n_weight*rules.shape[1], rules.shape[0]//3*n_height))
    
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = sns.light_palette('#F08080', 10)
    ax = sns.heatmap(rules, cmap=cmap, vmin=0, vmax=1, linewidths=1.5) #"YlGnBu"
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontweight = 'bold')
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches="tight")
    

if __name__ == '__main__':
    random_state = 1234
    args = arg_parser()

    if args.pickled:
        dataset = pickle.load(open(args.dataset, 'rb'))
    else:
        dataset = load_dataset.load_custom_data(args.dataset)

    # Read a list of trained classifiers from k-fold cross-validation.
    classifiers = pickle.load(open(args.model, 'rb'))

    if classifiers is not list:
        classifiers = [classifiers]

    for index in range(1):
        classifier = classifiers[index]
        save_folder = os.path.join('.', 'output', args.name)
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        save_name = os.path.join(save_folder, f'rep{index}')
        
        class Params(object):
            binary_pos_only = classifier.binary_pos_only
            n_rules = classifier.n_rules
            n_classes = classifier.n_classes
            epsilon = classifier.min_epsilon
            category_info = dataset.get('category_info')
            feature_names = dataset.get('feature_names')

        draw_rules_from_classifier(classifier, Params, keep_irrelevant_variable=True, 
                                       filter_redundant_rules=True,
                                       filter_similarity_threshold=0.7,
                                       save_prefix=save_name,
                                       show_encoding=True)
        