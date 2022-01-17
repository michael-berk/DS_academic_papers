# Description: potential implementation for IBM's FreaAI method
# Author: Michael Berk
# Date: Winter 2022

import pandas as pd
import numpy as np
import plotly.express as px

# XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# HDP
import scipy.stats.kde as kde
from matplotlib import pyplot as plt

# DT
import graphviz
from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
import itertools


#########################
# Step 1: load data and train XGBoost model
#########################
def read_data(show_df=False):
    """ Read, format, and return data.

        Data: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
        Data Docs: https://www.kaggle.com/uciml/pima-indians-diabetes-database
    """


    # read raw data
    df = pd.read_csv('DiabetesData/pima-indians-diabetes.data.csv', header=None)

    # add some more columns for fun (one hot encoded categorical)
    np.random.seed(0)
    enc_df = pd.DataFrame(dict(
        color=['red' if x > 0.25 else 'green' for x in np.random.rand(len(df.index))],
        gender=['male' if x > 0.55 else 'female' for x in np.random.rand(len(df.index))]
    ))

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(enc_df[['color','gender']]).toarray())

    df = pd.concat([df, enc_df], ignore_index=True, axis=1)

    df.columns = ['num_pregnancies','glucose','blood_pressure','skin_thickness','insulin','bmi','diabetes_pedigree','age','outcome', 'color_red','color_green','gender_male','gender_female']
    if show_df: print(df)

    return df

def train_baseline_model(df):
    """ Fit our baseline XGBoost model and return the OOS prediction EM bool for each val

        Src: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
    """

    # split data into X and y
    mask = np.array(list(df)) == 'outcome'
    X = df.loc[:,~mask]
    Y = df.loc[:,mask]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # add accuracy and return
    out = pd.concat([X_test, y_test], axis=1, ignore_index=True)
    out.columns = list(df) 

    accuracy_bool = (np.array(y_test).flatten() ==  np.array(predictions))
    out['accuracy_bool'] = accuracy_bool
    
    return out


#########################
# Step 2: find areas of weakness
#########################

#################### HDP ########################
def hpd_grid(sample, alpha=0.05, roundto=2, percent=0.5, show_plot=False):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    percent: float
        Perecent of data in the highest density region
    show_plots: bool
        if true, will show intermediary plots
    Returns
    ----------
    hpd: list with the highest density interval
    x: array with grid points where the density was evaluated
    y: array with the density values
    modes: list listing the values of the modes

    Src: https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py
    Note this was modified to find low-accuracy areas

    """

    # data points that create a density plot when histogramed
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    if show_plot: px.histogram(sample, title='Histogram of Bi-Modal Data', template='simple_white', color_discrete_sequence=['#177BCD']).show()

    # get upper and lower bounds on search space
    l = np.min(sample)
    u = np.max(sample)

    # get x-axis values
    x = np.linspace(l, u, 2000)

    # get kernel density estimate
    density = kde.gaussian_kde(sample)
    y = density.evaluate(x)

    if show_plot: px.scatter(x=x, y=y, title='Density Estimate of Bi-Modal Data', template='simple_white').update_traces(marker=dict(color='#177BCD')).show()

    # sort by size of y (density estimate), descending 
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)

    # get all x's where y is in the top 1-alpha percent
    # this is to bound the type 1 error
    xy_cum_sum = 0
    hdv = [] # x values
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break

    # determine horizontal line corresponding to percent 
    yy_zipped = zip(y, y/np.sum(y))
    yy = sorted(yy_zipped, key=lambda x: x[1], reverse=True)

    y_cum_sum = 0
    y_cutoff = 0
    for val in yy:
        y_cum_sum += val[1]
        if y_cum_sum >= percent:
            y_cutoff = val[0]
            break

    # get indices of sample in range 
    intersections = []
    for i, curr in enumerate(y):
        prior = y[i-1]
        if (prior < y_cutoff and curr >= y_cutoff) or (prior >= y_cutoff and curr < y_cutoff):
            intersections.append(x[i])

    indices = []
    for i in range(0, len(intersections), 2):
        lower, upper = intersections[i], intersections[i+1]
        indices.append([i for i,v in enumerate(sample) if v <= upper and v >= lower])

    # setup for difference comparison
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))

    # if y_i - y_{i-1} > diff then save
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))

    # prepare to calcualte value with highest density
    ite = iter(hpd)
    hpd = list(zip(ite, ite)) # create sequential pairs
    modes = []

    # find x and y value whith highest density
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto)) # store x-value where density is highest in range
    return hpd, x, y, modes, y_cutoff, np.array(indices).flatten()


def hpd_example(show_plot=False): 
    """ Plot an example of HDP method on bimodal data.

    Src: https://stackoverflow.com/questions/53671925/highest-density-interval-hdi-for-posterior-distribution-pystan
    """

    # include two modes
    samples = np.random.normal(loc=[-4,4], size=(1000, 2)).flatten()

    # compute high density regions
    hpd_mu, x_mu, y_mu, modes_mu, y_cutoff, indices = hpd_grid(samples, show_plot=False)

    plt.figure(figsize=(8,6))

    # raw data
    plt.hist(samples, density=True, bins=29, alpha=0.5)

    # estimated distribution
    plt.plot(x_mu, y_mu)
    plt.title('Highest Prior Density Region for Bi-Modal Data')

    # high density intervals
    for (x0, x1) in hpd_mu:
        plt.hlines(y=0, xmin=x0, xmax=x1, linewidth=5)
        plt.axvline(x=x0, color='grey', linestyle='--', linewidth=1)
        plt.axvline(x=x1, color='grey', linestyle='--', linewidth=1)

    # modes
    for xm in modes_mu:
        plt.axvline(x=xm, color='r')

    # 95% of data
    plt.axhline(y=y_cutoff, color='g')

    if show_plot: plt.show()

def hpd_iterative_search(col, accuracy, start_percent=0.5, end_percent=0.98, increment=0.05, acc_cutoff=0.01):
    """

    :param col: (pd.Series) univariate numeric col to search through
    :param accuracy: (pd.Series) boolean accuracy column
    :param start_percent: (flaot) percent to start with
    :param end_percent: (flaot) percent to end with
    :param increment: (float) value to increment by
    :param acc_cutoff: (float) accuracy cutoff to return data
    :return: (2d arry) of indices of problematic areas
    """

    out = {}
    
    prior_indices = {} 
    prior_acc = None
    prior_p = None 
    percents = np.arange(start_percent, end_percent, increment)[::-1] 

    # get smaller and smaller data slices
    for p in percents:
        # run HDP
        *_, indices = hpd_grid(col, percent=p)

        if indices.shape[0] != 0:

            # get accuracy for HDP
            indices = indices[0] if indices.shape[0] < 10 else indices
            acc = np.mean(accuracy.iloc[indices])

            # determine if there is a "meaningful" change - this is done with a stat sig cal
            if prior_acc is not None and acc - prior_acc > acc_cutoff:
                out[f'{p}-{prior_p}'] = (prior_indices - set(indices), acc - prior_acc, prior_acc)

            # reset
            prior_indices = set(indices)
            prior_acc = acc
            prior_p = p

    return out 

    

###################### Decision Tree #############
######### THIS IS INCOMPLETE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #####################
# Src: https://towardsdatascience.com/train-a-regression-model-using-a-decision-tree-70012c22bcc1
# Src: https://towardsdatascience.com/an-exhaustive-guide-to-classification-using-decision-trees-8d472e77223f
def fit_DT(df, predictors = ['age']):
    """ Fit a classification decision tree and return key elements """

    X = df[predictors] 
    y = df['accuracy_bool'] 

    model = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    tree_structure = model.tree_

    return model, preds, acc, tree_structure, X

def visualize_DT_1(df, model, predictors=['age'], fname='tree'):
    """ Visualize and export DT model """
    data = export_graphviz(model, feature_names=predictors,
                           filled=True, rounded=True, node_ids=True)

    graph = graphviz.Source(data)
    graph.render('trees/'+fname)

def visualize_DT_2(df, model, predictors=['age'], fname='tree'):
    """ Visualize tree with data plots """
    X = df[predictors]
    acc = df['accuracy_bool'].values
    viz = dtreeviz(model, X, acc,
                    target_name="accuracy_bool",
                    feature_names=predictors,
                    class_names=['True','False'] if acc[0] else ['False','True'])

    viz.save('trees/'+fname+'.svg')

def return_dt_split(model, col, accuracy, col_2=None, impurity_cutoff=0.4, n_datapoints_cutoff=5, acc_cutoff=0.03):
    """
    Return all indices of col that meet the following criteria:
    1. Leaf has accuracy lower that baseline by acc_cutoff
    2. Split size > n_datapoints_cutoff 

    :param model: SKLearn classification decision tree model
    :param col: (pd.Series) column used to split on
    :param accuracy: (pd.Series) column corresponding to correct/incorrect classification
    :param col_2: (pd.Series) column to be used for interactions
    :param impurity_cutoff: (float) requirement for entropy/gini of leaf
    :param n_datapoints_cutoff: (int) minimum n in a final node to be returned
    :param acc_cutoff: (float) accuracy cutoff for returning float
    :return: (dict[node_idx, indices]) where indices corresponds to the col that meet the above criteria
    """

    # get leaf ids and setup
    df = pd.concat([col, col_2], axis=1) if col_2 is not None else pd.DataFrame(col)
    leaf_id = model.apply(df)

    t = model.tree_
    baseline_acc = np.mean(accuracy)

    # get indices of leaf ids that meet criteria
    keeps = {i for i,v in enumerate(t.n_node_samples) if v > n_datapoints_cutoff} # sample size

    # store all data and corresponding index
    node_indices = {}
    for idx in keeps:
        node_indices[idx] = [i for i,v in enumerate(leaf_id) if v == idx]

        # remove non-low-accuracy areas and empty lists
        slice_acc = np.mean(accuracy.iloc[node_indices[idx]])
        if baseline_acc - slice_acc < acc_cutoff or node_indices[idx] == []:
            del node_indices[idx]

    return node_indices 

###################
# Run
###################

###################### Run Helpers #############
def run_data_search(df, viz=2):
    """ Iterate over data columns and perform the following actions...
    1. If type(col) is numeric, run HDP
    2. If type(col) is categorical, run DT
    3. Run DT for interactions
    Save DT viz and print problematic indices

    :param df: (pd.DataFrame) of raw data with correct/incorrect classification
    :param viz: (int) 0 if no viz, 1 if viz type 1, 2 if vis type 2
    """

    categoricals  = [x for x in list(df) if 'color' in x or 'gender' in x]
    acc_col = df['accuracy_bool']

    # univariate loop
    for col_name in [x for x in list(df) if x != 'accuracy_bool']:
        c = df[col_name]

        # numerics 
        if col_name not in categoricals and 'accuracy' not in col_name:
            print(f'Running: HDP for {col_name}')
            print(hpd_iterative_search(c, acc_col))

        # categoricals
        else:
            print(f'Running: DT for {col_name}')
            predictors = [col_name]
            model, *_, X = fit_DT(df, predictors)

            if viz == 1:
                visualize_DT_1(df, model, predictors=predictors, fname=f'{col_name}')
            elif viz == 2:
                visualize_DT_2(df, model, predictors=predictors, fname=f'{col_name}')

            print(return_dt_split(model, df['age'], acc_col))

    # bivariate loop (interactions)
    for col1, col2 in itertools.combinations(list(df), 2):
        print(f'Running: DT for {col1} and {col2}')
        predictors = [col1,col2]
        model, *_, X = fit_DT(df, predictors=predictors)

        if viz == 1:
            visualize_DT_1(df, model, predictors=predictors, fname=f'{col1}-{col2}')
        elif viz == 2:
            visualize_DT_2(df, model, predictors=predictors, fname=f'{col1}-{col2}')

        print(return_dt_split(model, df['age'], acc_col, df['bmi']))

###################### Run #############

# Step 1: train our baseline model 
df = read_data()
df_test = train_baseline_model(df) # add preds to df

# Step 2: run HPD Example
#hpd_example(show_plot=False)

# Step 3: run DT example
# Step 3.1: univariate
model, preds, acc, tree, X = fit_DT(df_test)
print(return_dt_split(model, df_test['age'], df_test['accuracy_bool']))

# Step 3.2: bivariate
predictors = ['age','bmi']
model, preds, acc, tree, X = fit_DT(df_test, predictors=predictors)
visualize_DT_2(df_test, model, predictors=predictors)
print(return_dt_split(model, df_test['age'], df_test['accuracy_bool'], df_test['bmi']))

# Step 4: find areas of weakness
run_data_search(df_test)
