import pandas as pd
import numpy as np
import shap
from numpy import argmax
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, fbeta_score, balanced_accuracy_score, log_loss, brier_score_loss, roc_auc_score, roc_curve, auc
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
import tune
import optuna
import hyperopt
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

def standardise_data(X_train, X_test):
    """
    Converts all data to a similar scale.
    Standardisation subtracts mean and divides by standard deviation
    for each feature.
    Standardised data will have a mena of 0 and standard deviation of 1.
    The training data mean and standard deviation is used to standardise both
    training and test set data.
    """
    
    # Initialise a new scaling object for normalising input data
    sc = StandardScaler() 

    # Set up the scaler just on the training set
    sc.fit(X_train)

    # Apply the scaler to the training and test sets
    train_std=sc.transform(X_train)
    test_std=sc.transform(X_test)
    
    return train_std, test_std


def calculate_accuracy(observed, predicted):
    
    """
    Calculates a range of accuracy scores from observed and predicted classes.
    
    Takes two list or NumPy arrays (observed class values, and predicted class 
    values), and returns a dictionary of results.
    
     1) observed positive rate: proportion of observed cases that are +ve
     2) Predicted positive rate: proportion of predicted cases that are +ve
     3) observed negative rate: proportion of observed cases that are -ve
     4) Predicted negative rate: proportion of predicted cases that are -ve  
     5) accuracy: proportion of predicted results that are correct    
     6) precision: proportion of predicted +ve that are correct
     7) recall: proportion of true +ve correctly identified
     8) f1: harmonic mean of precision and recall
     9) sensitivity: Same as recall
    10) specificity: Proportion of true -ve identified:        
    11) positive likelihood: increased probability of true +ve if test +ve
    12) negative likelihood: reduced probability of true +ve if test -ve
    13) false positive rate: proportion of false +ves in true -ve patients
    14) false negative rate: proportion of false -ves in true +ve patients
    15) true positive rate: Same as recall
    16) true negative rate
    17) positive predictive value: chance of true +ve if test +ve
    18) negative predictive value: chance of true -ve if test -ve
    
    """
    
    # Converts list to NumPy arrays
    if type(observed) == list:
        observed = np.array(observed)
    if type(predicted) == list:
        predicted = np.array(predicted)
    
    # Calculate accuracy scores
    observed_positives = observed == 1
    observed_negatives = observed == 0
    predicted_positives = predicted == 1
    predicted_negatives = predicted == 0
    
    true_positives = (predicted_positives == 1) & (observed_positives == 1)
    
    false_positives = (predicted_positives == 1) & (observed_positives == 0)
    
    true_negatives = (predicted_negatives == 1) & (observed_negatives == 1)
    
    false_negatives = (predicted_negatives == 1) & (observed_negatives == 0)
    
    accuracy = np.mean(predicted == observed)
    
    precision = (np.sum(true_positives) /
                 (np.sum(true_positives) + np.sum(false_positives)))
        
    recall = np.sum(true_positives) / np.sum(observed_positives)
    
    sensitivity = recall
    
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    specificity = np.sum(true_negatives) / np.sum(observed_negatives)
    
    positive_likelihood = sensitivity / (1 - specificity)
    
    negative_likelihood = (1 - sensitivity) / specificity
    
    false_positive_rate = 1 - specificity
    
    false_negative_rate = 1 - sensitivity
    
    true_positive_rate = sensitivity
    
    true_negative_rate = specificity
    
    positive_predictive_value = (np.sum(true_positives) / 
                                 np.sum(observed_positives))
    
    negative_predictive_value = (np.sum(true_negatives) / 
                                  np.sum(observed_negatives))
    
    # Create dictionary for results, and add results
    results = dict()
    
    results['observed_positive_rate'] = np.mean(observed_positives)
    results['observed_negative_rate'] = np.mean(observed_negatives)
    results['predicted_positive_rate'] = np.mean(predicted_positives)
    results['predicted_negative_rate'] = np.mean(predicted_negatives)
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['sensitivity'] = sensitivity
    results['specificity'] = specificity
    results['positive_likelihood'] = positive_likelihood
    results['negative_likelihood'] = negative_likelihood
    results['false_positive_rate'] = false_positive_rate
    results['false_negative_rate'] = false_negative_rate
    results['true_positive_rate'] = true_positive_rate
    results['true_negative_rate'] = true_negative_rate
    results['positive_predictive_value'] = positive_predictive_value
    results['negative_predictive_value'] = negative_predictive_value
    
    return results

def train_xgb(X_train, y_train):
    '''
    Trains an xgboost algorithm using optuna bayesian hyperoptimisation. The HPs are in the tune.py folder
    '''
    # Set up and fit model (n_jobs=-1 uses all cores on a computer)

    #pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: tune.xgb_objective(trial, X_train, y_train), n_trials=50)#100? 
    #study.optimize(lambda trial: tune.objective_trial(trial, X_train, y_train), n_trials=50)
    # Clears the output after each fold to laptop doesn't crash
    clear_output()
    best_params = study.best_params
    model = XGBClassifier(**best_params, eval_metrics='logloss')
    
    #model = XGBClassifier(n_jobs=-1)
    model.fit(X_train,y_train)
    #model = CalibratedClassifierCV(model)
    #model.fit(X_train, y_train)
    return model

def train_lr(X_train, y_train):
    '''
    Trains a logistic regression model using hyperopt for bayesian hyperparameter tuning
    '''
    # Define hyperparameter space
    params = {'penalty' : hyperopt.hp.choice('penalty', ['l2', 'l1']),
              'C' : hyperopt.hp.loguniform('C', -4, 4),
              'solver' : hyperopt.hp.choice('solver', ['lbfgs', 'liblinear']),
              'random_state': 42
             }
    # Tune using hyperopt
    best_params = tune.hyperopt_tune(LogisticRegression, params, None, X_train,
                                             y_train, 'roc_auc', 60)
    # Fit logistic regression with params from hyperopt
    model = LogisticRegression(**best_params)
    model.fit(X_train,y_train)
    return model

def k_fold_accuracies(X, y, strat_col, lr=True, smote=False):
    '''
    This function does a lot
    '''
    X = pd.get_dummies(X)
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_np = X.values
    y_np = y.values
    
    # Set up k-fold training/test splits
    number_of_splits = 10
    skf = StratifiedKFold(n_splits = number_of_splits, shuffle=True, random_state=42)
    splits = skf.split(X_np, strat_col)

    # Set up thresholds
    thresholds = np.arange(0, 1.01, 0.01)

    # Create arrays for overall results (rows=threshold, columns=k fold replicate)
    results_accuracy = np.zeros((len(thresholds),number_of_splits))
    results_precision = np.zeros((len(thresholds),number_of_splits))
    results_recall = np.zeros((len(thresholds),number_of_splits))
    results_f1 = np.zeros((len(thresholds),number_of_splits))
    results_predicted_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_observed_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_true_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_false_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_auc = []
    results_brier = []
    results_logloss =[]
    results_specificity = np.zeros((len(thresholds),number_of_splits))
    results_balanced_accuracy = np.zeros((len(thresholds),number_of_splits))

    # Test and predicted for each fold
    test_sets = []
    predicted_probas = []
    
    # Loop through the k-fold splits
    loop_index = 0
    for train_index, test_index in splits:

        # Create lists for k-fold results
        threshold_accuracy = []
        threshold_precision = []
        threshold_recall = []
        threshold_f1 = []
        threshold_predicted_positive_rate = []
        threshold_observed_positive_rate = []
        threshold_true_positive_rate = []
        threshold_false_positive_rate = []
        threshold_specificity = []
        threshold_balanced_accuracy = []

        # Get X and Y train/test
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        # To smote or not to smote
        if smote:
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        # Set up and fit model (n_jobs=-1 uses all cores on a computer)
        if lr:
            model = train_lr(X_train, y_train)
        else:
            model = train_xgb(X_train, y_train)

        # Get probability of hypo and non-hypo
        probabilities = model.predict_proba(X_test)
        # Take just the hypo probabilities (column 1)
        predicted_proba = probabilities[:,1]
        predicted_probas.append(predicted_proba)
        # Add probabilities to test set for future analysis
        test_sets.append(test_index)
        
        # Loop through increments in probability of survival
        for cutoff in thresholds: #  loop 0 --> 1 on steps of 0.1
            # Get whether passengers survive using cutoff
            predicted_survived = predicted_proba >= cutoff
            # Call accuracy measures function
            accuracy = calculate_accuracy(y_test, predicted_survived)
            # Add accuracy scores to lists
            threshold_accuracy.append(accuracy['accuracy'])
            threshold_precision.append(accuracy['precision'])
            threshold_recall.append(accuracy['recall'])
            threshold_f1.append(accuracy['f1'])
            threshold_predicted_positive_rate.append(
                    accuracy['predicted_positive_rate'])
            threshold_observed_positive_rate.append(
                    accuracy['observed_positive_rate'])
            threshold_true_positive_rate.append(accuracy['true_positive_rate'])
            threshold_false_positive_rate.append(accuracy['false_positive_rate'])
            threshold_specificity.append(accuracy['specificity'])
            threshold_balanced_accuracy.append((accuracy['specificity']+accuracy['recall'])/2)

        # Add results to results arrays
        results_accuracy[:,loop_index] = threshold_accuracy
        results_precision[:, loop_index] = threshold_precision
        results_recall[:, loop_index] = threshold_recall
        results_f1[:, loop_index] = threshold_f1
        results_predicted_positive_rate[:, loop_index] = \
            threshold_predicted_positive_rate
        results_observed_positive_rate[:, loop_index] = \
            threshold_observed_positive_rate
        results_true_positive_rate[:, loop_index] = threshold_true_positive_rate
        results_false_positive_rate[:, loop_index] = threshold_false_positive_rate
        results_specificity[:, loop_index] = threshold_specificity
        results_balanced_accuracy[:, loop_index] = threshold_balanced_accuracy
        
        # Calculate ROC AUC
        roc_auc = auc(threshold_false_positive_rate, threshold_true_positive_rate)
        results_auc.append(roc_auc)
        
        # Calculate brier and logloss
        brier = brier_score_loss(y_test, predicted_proba)
        results_brier.append(brier)
        logloss = log_loss(y_test, predicted_proba)
        results_logloss.append(logloss)
        
        # Increment loop index
        loop_index += 1
    
     # Transfer summary results to dataframe
    results = pd.DataFrame(thresholds, columns=['thresholds'])
    results['accuracy'] = results_accuracy.mean(axis=1)
    results['accuracy_std'] = results_accuracy.std(axis=1)
    
    results['precision'] = results_precision.mean(axis=1)
    results['precision_std'] = results_precision.std(axis=1)
    
    results['recall'] = results_recall.mean(axis=1)
    results['recall_std'] = results_recall.std(axis=1)

    results['f1'] = results_f1.mean(axis=1)
    results['f1_std'] = results_f1.std(axis=1)
    
    results['predicted_positive_rate'] = \
        results_predicted_positive_rate.mean(axis=1)
    results['observed_positive_rate'] = \
        results_observed_positive_rate.mean(axis=1)
    
    results['true_positive_rate'] = results_true_positive_rate.mean(axis=1)
    results['false_positive_rate'] = results_false_positive_rate.mean(axis=1)
    
    results['specificity']= results_specificity.mean(axis=1)
    results['specificity_std']= results_specificity.std(axis=1)
    
    results['balanced_accuracy']= results_balanced_accuracy.mean(axis=1)
    results['balanced_accuracy_std']= results_balanced_accuracy.std(axis=1)
    
    results['roc_auc'] = np.mean(results_auc)
    results['roc_auc_std'] = np.std(results_auc)
    
    # Select threshold with the best balance between sensitivity and specificity
    ix = argmax(results_balanced_accuracy.mean(axis=1))
    
    # Add results at this threshold to k-fold results
    k_fold_results = [results_auc, results_brier, results_logloss, results_accuracy[ix], results_precision[ix], results_recall[ix], results_f1[ix], results_predicted_positive_rate[ix], results_observed_positive_rate[ix], results_true_positive_rate[ix], results_false_positive_rate[ix], results_specificity[ix], results_balanced_accuracy[ix]]
    k_fold_results = pd.DataFrame(k_fold_results).T
    k_fold_results.columns=['roc', 'brier', 'logloss','accuracy', 'precision', 'recall', 'f1','predicted_positive_rate', 'observed_positive_rate', 'tpr','fpr','specificity','balanced_accuracy']
    
    return k_fold_results, test_sets, predicted_probas, results

def add_mean_to_df(mean_df, k_fold_results, model_name, feature_set):
    '''
    Adds the mean of the k-fold results to a df and returns the modified df
    '''
    mean = k_fold_results.describe().loc['mean']
    mean['model'] = model_name
    mean['features'] = feature_set
    mean_df = mean_df.append(mean)
    return mean_df

def add_proba_col(df, test_sets_index, predicted_probas, colname):
    '''
    Adds a column to existing df with features with the probabilties of the folds
    '''
    # Initiate new column in dataframe
    df[[colname, colname+'_fold']] = -1
    for i in range(0,len(test_sets_index)):
        df[colname].iloc[test_sets_index[i]] = predicted_probas[i]
        df[colname+'_fold'].iloc[test_sets_index[i]] = i
    return df

def create_contour_plot(X, y, params, minim, maxim, title, smote=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    if smote:
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    X_train_std, X_test_std = standardise_data(X_train, X_test)
    
    model = LogisticRegression
    best_params = tune.hyperopt_tune(model, params, None, X_train_std, y_train, 'roc_auc', 60)
    tuned_model = model(**best_params)
    #tuned_model = LogisticRegression()

    tuned_model.fit(X_train_std,y_train)

    # Predict training and test set labels
    y_pred_train = tuned_model.predict(X_train_std)
    y_pred_test = tuned_model.predict(X_test_std)
    
    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)

    print ('Accuracy of predicting training data =', accuracy_train)
    print ('Accuracy of predicting test data =', accuracy_test)
    # After doing the contour plot on full extent values, I reduced age to ages that made a difference in the plot
    x1 = np.linspace(4,15,100)
    x2 = np.linspace(minim,maxim,100)

    #x2 = np.linspace(10,120,20)
    
    start_glc = []
    duration = []
    for i in x1:
        for j in x2:
            start_glc.append(i)
            duration.append(j)
    test_df = pd.DataFrame()
    test_df['Starting glucose'] = start_glc
    test_df['Duration'] = duration
    '''
    X_train_std, X_test_std = standardise_data(X_train, test_df)
    '''
    y = tuned_model.predict_proba(X_test_std)
    # Get probability of survival
    y = y[:, 1].flatten()
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    y_grid_hypo = np.reshape(y, x1_grid.shape)
    
    ## Create figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    cs = ax.contourf(x2_grid, x1_grid, y_grid_hypo.T, cmap='jet', levels=100)

    ax.set_ylabel('Starting glucose (mmol/L)')
    ax.set_xlabel('Duration of exercise (mins)')
    ax.set_title(title)
    # Add a colorbar
    cbar = plt.colorbar(cs, shrink=0.8, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    cbar.ax.set_ylabel('Risk of hypoglycaemia')
    ax.grid()
    return fig

def create_line_graph(ax, X, y, params, minim, maxim, title, smote=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    if smote:
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    X_train_std, X_test_std = standardise_data(X_train, X_test)
    

    model = LogisticRegression
    best_params = tune.hyperopt_tune(model, params, None, X_train_std, y_train, 'roc_auc', 60)
    tuned_model = model(**best_params)
    #tuned_model = LogisticRegression()

    tuned_model.fit(X_train_std,y_train)

    # Predict training and test set labels
    y_pred_train = tuned_model.predict(X_train_std)
    y_pred_test = tuned_model.predict(X_test_std)
    
    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)

    print ('Accuracy of predicting training data =', accuracy_train)
    print ('Accuracy of predicting test data =', accuracy_test)
    # After doing the contour plot on full extent values, I reduced age to ages that made a difference in the plot
    x1 = np.linspace(minim, maxim,100)
    
    start_glc = []
    for i in x1:
        start_glc.append(i)
    test_df = pd.DataFrame()
    test_df['Starting glucose'] = start_glc
    X_train_std, X_test_std = standardise_data(X_train, test_df)
    y = tuned_model.predict_proba(X_test_std)
    # Get probability of survival
    y = y[:, 1].flatten()
    #x1_grid, x2_grid = np.meshgrid(x1, x2)
    #y_grid_hypo = np.reshape(y, x1_grid.shape)
    
    # Create figure
    #fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(start_glc, y)
    ax.set_title(title)
    ax.set_xlabel('Finishing glucose (mmol/L)')
    ax.set_ylabel('Risk of hypoglycaemia (<3.9mmol/L)')

    return ax


def run_k_fold_model(X, y, model, params, strat_col, fit_params=False, metric='roc_auc', number_of_splits=10, smote=False, tune_hp=True):
    # Set up lists to hold results for each k-fold run
    avg_results = []
    coeffs = []
    # Set up lists for observed and predicted
    test_indices = []
    observed = []
    predicted_proba = []

    # Set up splits

    skf = StratifiedKFold(n_splits = number_of_splits, shuffle=True, random_state=42)
    splits = skf.split(X, strat_col)
    
    # Loop through the k-fold splits
    for train_index, test_index in splits:
        # Get X and Y train/test
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test = standardise_data(X_train, X_test)
        if smote:
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        fit_param_space=None
        if tune_hp:
            best_params = tune.hyperopt_tune(model, params, fit_param_space, X_train,
                                             y_train, metric, 60)
            tuned_model = model(**best_params)
        else:
            tuned_model = model()
        tuned_model.fit(X_train, y_train)
        y_probs = tuned_model.predict_proba(X_test)[:,1]
        observed.append(y_test)
        predicted_proba.append(y_probs)
        
        # explain the model's predictions
        # Coeffs
        coeffs.append(tuned_model.coef_[0])

    return observed, predicted_proba, coeffs

def calculate_coefficient(coeffs, column_names):

    log_weights = pd.DataFrame(coeffs, columns=['weights'], index=column_names)
    log_weights['abs_weight'] = abs(log_weights['weights'])
    log_weights.sort_values('abs_weight', ascending=False, inplace=True)
    # Set up figure
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    # Get labels and values
    labels = log_weights.index.values[0:25]
    pos = np.arange(len(labels))
    val = log_weights['weights'].values[0:25]

    # Plot
    ax.bar(pos, val)
    ax.set_ylabel('Feature weight (standardised features)')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    plt.suptitle('Weights of features')
    plt.tight_layout()
    #plt.savefig('output/lr_single_fit_feature_weights_bar.jpg', dpi=300)
    return log_weights, fig