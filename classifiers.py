'''
Machine Learning Pipeline

Angelica Valdiviezo

TODO
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
from datetime import datetime
from scipy import stats
from data_processing import get_outcome_lbl, get_projects_df, get_cols_to_discrete

# TODO delete
import pytz
TZ_CHI = pytz.timezone('America/Chicago') 

#classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

#metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
                            recall_score, precision_recall_curve,\
                            confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import tree

SEED = 0
T_SIZE = 0.1
VERBOSE = False

###############################################################################
############################# 1) BUILD AND TRAIN CLASSIFIERS ##################
###############################################################################

def run_clfs(grid_size, thresholds, temporal_dfs, plot=False, write_csv=False):
    '''
    TODO
    '''
    dT_CHI = datetime.now(TZ_CHI)
    time_now = dT_CHI.strftime("%H_%M_%S")
    print("Start time:", time_now)

    clfs, grid = get_clfs_params(grid_size)
    results = get_results_df(thresholds)
    outcome = get_outcome_lbl()
    if write_csv:
        csv_file = 'results/{}grid_{}_results_writing.csv'.format(grid_size, time_now)
        f = open(csv_file, 'w')

    # TODO: DELETE
    n = 1

    for split_date in temporal_dfs:
        # Get test df and its dates
        test_df = temporal_dfs[split_date]['test']['df']
        test_start = temporal_dfs[split_date]['test']['start_date']
        test_end = temporal_dfs[split_date]['test']['end_date']

        # Get train df and its dates
        train_df = temporal_dfs[split_date]['train']['df']
        train_start = temporal_dfs[split_date]['train']['start_date']
        train_end = temporal_dfs[split_date]['train']['end_date']

        # Get features and labels for test and train dfs
        predictors = get_predictors(test_df) # they are the same for all
        X_test = test_df[predictors]
        y_test = test_df[outcome]
        X_train = train_df[predictors]
        y_train = train_df[outcome]

        # Fit each of the models with the training data
        for clf in clfs:
            print()
            print()
            print("Working with: ", clf)
            params = grid[clf]
            model = clfs[clf]

            # Create a Param Grid with the params to explore
            for p in ParameterGrid(params):
                try:
                    print()
                    print("\tModel", n)
                    n += 1
                    model.set_params(**p)
                    print("\t\tModel", model)
                    model.fit(X_train, y_train)
                    print('\t\tSuccesfully trained model.')

                    # Apply the model to the test data
                    if clf == 'SVM':
                        y_pred_probs = model.decision_function(X_test)
                    else:
                        y_pred_probs = model.predict_proba(X_test)[:,1]
                    print('\t\tSuccesfully predicted.')

                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(
                            y_pred_probs, y_test), reverse=True))

                    # Calculate the performance metrics for the model
                    metrics_stats = []
                    for thres in thresholds:
                        pres = precision_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                        rec = recall_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                        f1 = f1_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                        metrics_stats.extend([pres, rec, f1])

                    # Create the row that will be inserted in the results df with the info for this model
                    # The position for each row element corresponds exactly to the colums in the results df
                    model_info = [clf, model, p, split_date,
                                  train_start, train_end, test_start, test_end,
                                  precision_at_k(y_test_sorted, y_pred_probs_sorted, 100)] +\
                                  metrics_stats +\
                                  [roc_auc_score(y_test, y_pred_probs)]

                    # Add the model_info row to the next row in the results df
                    if not write_csv:
                        results.loc[len(results)] = model_info
                        print("\t\tAdded metrics row in results.")
                    else:
                        print("trying to save in csv")
                        model_row = (",").join(model_info)
                        f.write(model_row + '\n')

                    if plot:
                        plot_precision_recall_n(y_test, y_pred_probs, clf)

                        #Plot histogram of predicted scores
                        plt.hist(y_pred_probs)
                        plt.title('Histogram of Yscores')
                        plt.show()

                        # Plot features' importance
                        get_feature_importance(clf, X_train, model)

                except:
                    print('An error happened')
                    continue

    dT_CHI = datetime.now(TZ_CHI)
    time_now = dT_CHI.strftime("%H_%M_%S")
    file_name = 'results/grid_{}_time_{}results.csv'.format(grid_size, time_now)
    if not write_csv:
        try:
            results.to_csv(file_name)
            print("Saved file:", file_name)
        except:
            print('could not save file: ', file_name)
    else:
        f.close()

    return results

def run_clfs_RFs_test(clfs, grid, thresholds, temporal_dfs, plot=False, save_csv=False):
    '''
    TODO
    '''
    dT_CHI = datetime.now(TZ_CHI)
    time_now = dT_CHI.strftime("%H_%M_%S")
    print("Start time:", time_now)

    results = get_results_df(thresholds)
    outcome = get_outcome_lbl()

    # TODO: DELETE
    n = 1

    for split_date in temporal_dfs:
        # Get test df and its dates
        test_df = temporal_dfs[split_date]['test']['df']
        test_start = temporal_dfs[split_date]['test']['start_date']
        test_end = temporal_dfs[split_date]['test']['end_date']

        # Get train df and its dates
        train_df = temporal_dfs[split_date]['train']['df']
        train_start = temporal_dfs[split_date]['train']['start_date']
        train_end = temporal_dfs[split_date]['train']['end_date']

        # Get features and labels for test and train dfs
        predictors = get_predictors(test_df) # they are the same for all
        X_test = test_df[predictors]
        y_test = test_df[outcome]
        X_train = train_df[predictors]
        y_train = train_df[outcome]

        # Fit each of the models with the training data
        for clf in clfs:
            print()
            print()
            print("Working with: ", clf)
            params = grid[clf]
            model = clfs[clf]

            # Create a Param Grid with the params to explore
            for p in ParameterGrid(params):
                try:
                    print()
                    print("\tModel", n)
                    n += 1
                    model.set_params(**p)
                    print("\t\tModel", model)
                    model.fit(X_train, y_train)
                    print('\t\tSuccesfully trained model.')

                    # Apply the model to the test data
                    if clf == 'SVM':
                        y_pred_probs = model.decision_function(X_test)
                    else:
                        y_pred_probs = model.predict_proba(X_test)[:,1]
                    print('\t\tSuccesfully predicted.')

                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(
                            y_pred_probs, y_test), reverse=True))

                    # Calculate the performance metrics for the model
                    metrics_stats = []
                    for thres in thresholds:
                        pres = precision_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                        rec = recall_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                        f1 = f1_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                        metrics_stats.extend([pres, rec, f1])

                    # Create the row that will be inserted in the results df with the info for this model
                    # The position for each row element corresponds exactly to the colums in the results df
                    model_info = [clf, model, p, split_date,
                                  train_start, train_end, test_start, test_end,
                                  precision_at_k(y_test_sorted, y_pred_probs_sorted, 100)] +\
                                  metrics_stats +\
                                  [roc_auc_score(y_test, y_pred_probs)]

                    # Add the model_info row to the next row in the results df
                    results.loc[len(results)] = model_info
                    print("\t\tAdded metrics row in results.")

                    if plot:
                        plot_precision_recall_n(y_test, y_pred_probs, clf)

                        #Plot histogram of predicted scores
                        plt.hist(y_pred_probs)
                        plt.title('Histogram of Yscores')
                        plt.show()

                        # Plot features' importance
                        get_feature_importance(clf, X_train, model)

                except:
                    print('An error happened')
                    continue

    dT_CHI = datetime.now(TZ_CHI)
    time_now = dT_CHI.strftime("%H_%M_%S")
    file_name = 'results/grid_{}_time_{}results.csv'.format('RF', time_now)
    if save_csv:
        try:
            results.to_csv(file_name)
            print("Saved file:", file_name)
        except:
            print('could not save file: ', file_name)

    return results



### HELPERS ###
def get_clfs_params(grid_size):
    '''
    Get the parameters for a given grid size: test, small, medium, large.
    Inputs:
        -grid_size(str): 'test', 'small', 'med', 'large'

    Ouputs:
        -a dictionary that maps the type of a classifier to an instance
            of that classifier
        -a dictionary that maps the type of a classifier to a dictionary that
            maps name of params to param values.
    Note: this function is based on code for the group project with Chi Nguyen and Camilo Arias.
    Repo: https://github.com/cariasmartelo/eviction-data-quality
    '''
    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'B': BaggingClassifier(),
            'LR': LogisticRegression(penalty='l1', C=1e5, solver='liblinear'),
            'SVM': LinearSVC(),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'DT': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1) 
            }

    test_grid = { 
        'RF':   {'n_estimators': [1], 
                 'max_depth': [1], 
                 'max_features': ['sqrt'],
                 'min_samples_split': [10], 
                 'n_jobs': [-1]},
        'B':    {'n_estimators': [1]},
        'LR':   {'penalty': ['l1'], 
                 'C': [0.01]},
        'SVM':  {'C': [0.01]},
        'GB':   {'n_estimators': [1], 
                 'learning_rate' : [0.1],
                 'subsample' : [0.5], 
                 'max_depth': [1]},
        'DT':   {'criterion': ['gini'], 
                 'max_depth': [1], 
                 'max_features': [None],
                 'min_samples_split': [10]},
        'KNN':  {'n_neighbors': [5],
                 'weights': ['uniform'],
                 'algorithm': ['auto']}
                }

    small_grid = { 
        'RF':   {'n_estimators': [100, 10000], 
                  'max_depth': [5,50], 
                  'max_features': ['sqrt','log2'],
                  'min_samples_split': [2,10], 
                  'n_jobs':[-1]},
        'B':    {'n_estimators': [1,10,100,1000,10000]},
        'LR':   {'penalty': ['l1','l2'], 
                 'C': [0.00001,0.001,0.1,1,10]},
        'SVM':  {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'GB':   {'n_estimators': [100, 1000], 
                 'learning_rate' : [0.001,0.1,0.5],
                 'subsample' : [0.1,1.0], 
                 'max_depth': [5,50]},
        'DT':   {'criterion': ['gini', 'entropy'], 
                 'max_depth': [1,5,10,20,50,100], 
                 'max_features': [None,'sqrt','log2'],
                 'min_samples_split': [2,5,10]},
        'KNN':  {'n_neighbors': [5,10],
                 'weights': ['uniform'],
                 'algorithm': ['auto']}
                 }

    med_grid = { 
        'RF':   {'n_estimators': [1, 10, 100, 1000], 
                 'max_depth': [5,50], 
                 'max_features': ['sqrt','log2'],
                 'min_samples_split': [2,10], 
                 'n_jobs':[-1]},
        'B':    {'n_estimators': [1,10,100,1000]},
        'LR':   {'penalty': ['l1','l2'], 
                 'C': [0.00001,0.001,0.1,1,10]},
        'SVM':  {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'GB':   {'n_estimators': [1, 50], 
                 'learning_rate' : [0.1, 0.5],
                 'subsample' : [0.1, 1.0], 
                 'max_depth': [1, 5, 10]},
        'DT':   {'criterion': ['gini', 'entropy'], 
                 'max_depth': [1,5,10,20,50,100], 
                 'max_features': [None,'sqrt','log2'],
                 'min_samples_split': [2,5,10]},
        'KNN':  {'n_neighbors': [5],
                 'weights': ['uniform'],
                 'algorithm': ['auto']}
               }

    large_grid = { 
        'RF':   {'n_estimators': [1,10,100,1000,10000], 
                 'max_depth': [1,5,10,20,50,100], 
                 'max_features': ['sqrt','log2'],
                 'min_samples_split': [2,5,10], 
                 'n_jobs': [-1]},
        'B':    {'n_estimators': [1,10,100,1000,10000]},
        'LR':   {'penalty': ['l1','l2'], 
                 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'SVM':  {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'GB':   {'n_estimators': [1,10,100,1000,10000], 
                 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
                 'subsample' : [0.1,0.5,1.0],
                 'max_depth': [1,3,5,10,20,50,100]},
        'DT':   {'criterion': ['gini', 'entropy'],
                 'max_depth': [1,5,10,20,50,100],
                 'max_features': [None, 'sqrt','log2'],
                 'min_samples_split': [2,5,10]},
        'KNN':  {'n_neighbors': [1,5,10,25,50,100],
                 'weights': ['uniform','distance'],
                 'algorithm': ['auto','ball_tree','kd_tree']}
                  }

    if grid_size == 'large':
        return clfs, large_grid
    elif grid_size == 'small':
        return clfs, small_grid
    elif grid_size == 'test':
        return clfs, test_grid
    elif grid_size == 'med':
        return clfs, med_grid
    else:
        return 0, 0

def get_results_df(thresholds, metrics=None):
    '''
    Returns an empty DF which will store the information about the models
    run as well their results.

    Inputs:
        - thresholds(list): list of thresholds
        - metrics: evaluation metrics. Default are precision, recall and f1.
    '''
    if not metrics:
        metrics = ['p_at_', 'recall_at_', 'f1_at_']

    # Making the metrics at the different thresholds
    metrics_at = []
    for threshold in thresholds:
        for metric in metrics:
            metrics_at.append(metric + str(threshold) + 'pct')

    info_cols = ['model_type', 'clf_details', 'chosen_params', 'split_date'] + \
                ['start_date_train', 'end_date_train', 'start_date_test', 'end_date_test'] +\
                ['random_baseline'] +\
                metrics_at + ['auc-roc']

    results_df = pd.DataFrame(columns=info_cols)

    return results_df

def get_predictors(df):
    '''
    TODO
    '''
    predictors = list(df.loc[:,df.apply(lambda x: x.isin([0, 1]).all())].columns)
    outcome = get_outcome_lbl()
    try:
        predictors.remove(outcome)
    except ValueError:
        pass

    return predictors

def precision_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def f1_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1

def generate_binary_at_k(y_scores, k):
    '''
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    HUGE TODO - credit to curve in prof. rayid's material for the class
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()

def get_feature_importance(model_name, X_train, clf, n_importances=10):
    '''
    This function returns the feature importances list
    '''
    if model_name in ['DT', 'RF', 'GB']:
        importances = clf.feature_importances_
    elif model_name == 'B':
        importances = np.mean([tree.feature_importances_ for\
                               tree in clf.estimators_], axis=0)
    elif model_name == 'LR':
        importances = abs(clf.coef_[0]) * np.array(np.std(X_train, 0))
        importances = (importances / importances.max())
    else:
        return

    indices = np.argsort(importances)[::-1]

    print('FEATURE IMPORTANCES')
    print()
    f_importances = []
    important_features = []
    for f in range(n_importances):
        if importances[indices[f]] > 0:
            important_features.append(X_train.columns[indices[f]])
            f_importances.append(importances[indices[f]])
            print("%d. Feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))
    plt.clf()
    fig, ax = plt.subplots()
    ys = np.arange(len(important_features))
    xs = np.array(f_importances)
    ax.barh(ys, xs)
    ax.set_yticks(ys) #Replace default x-ticks with xs, then replace xs with labels
    ax.set_yticklabels(important_features) #Replace default x-ticks with xs, then replace xs with labels
    ax.invert_yaxis()
    ax.set_title('Feature Importance')
    plt.shoow()

def joint_sort_descending(l1, l2):
    '''
    Inputs:
        - l1: numpy array
        - l2: numpy array
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

###############################################################################
############################### 2) SAVE PREDICTIONS ###########################
###############################################################################

def get_predictions(clf, train_df, test_df, threshold=5, save_csv=True):
    '''
    Trains and predicts for a given classifier and returns a df with
    a 'prediction' column.

    Input:
        - clf: classifier object
        - train_df and test_df
        - threshold: to get top k
        - save_csv (bool): Default true, to save csv file with prediction
            results

    Output: results df, which includes all IDs and numeric columns that
        where discretized and dropped
    '''
    predictors = get_predictors(test_df)
    outcome = get_outcome_lbl()
    X_test = test_df[predictors]
    X_train = train_df[predictors]
    y_train = train_df[outcome]

    # Train
    clf.fit(X_train, y_train)

    # Get prediction probabilities
    if isinstance(clf, LinearSVC):
        y_pred_probs = clf.decision_function(X_test)
    else:
        y_pred_probs = clf.predict_proba(X_test)[:,1]
    X_test['pred_prob'] = y_pred_probs

    # Get binary prediction
    X_test.sort_values(by='pred_prob', axis=0, ascending=False, inplace=True)
    y_prediction = generate_binary_at_k(y_pred_probs, threshold)
    X_test['prediction'] = y_prediction
    X_test.drop('pred_prob', axis=1, inplace=True)

    # Get ids from test_df
    original_df = get_projects_df()
    numeric_cols = get_cols_to_discrete(original_df)
    id_cols = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid']
    results = X_test.join(original_df[id_cols + numeric_cols])

    if save_csv:
        file_name = 'results/prediction_results.csv'
        results.to_csv(file_name)

    return results





