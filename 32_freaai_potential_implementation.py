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

#########################
# Step 1: load data and train XGBoost model
#########################
def read_data():
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
    print(df)

    return df

def train_baseline_model(df):
    """ Fit our baseline XGBoost model 

        Src: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
    """

    # split data into X and y
    mask = np.array(list(df)) == 'outcome'
    X = df.loc[:,~mask].to_numpy()
    Y = df.loc[:,mask].to_numpy()

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


#########################
# Step 2: find areas of weakness
#########################

#################### HDP ########################
def hpd_grid(sample, alpha=0.05, roundto=2):
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
    Returns
    ----------
    hpd: list with the highest density interval
    x: array with grid points where the density was evaluated
    y: array with the density values
    modes: list listing the values of the modes

    Src: https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py
    Note this was modified to find low-accuracy areas

    """

    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]

    # get upper and lower bounds on search space
    l = np.min(sample)
    u = np.max(sample)

    # get kernel density estimate
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    print(list(x))
    y = density.evaluate(x)
    print(list(y))
    px.scatter(x=x, y=y).show()

    # loop through density curve
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)

    # sum to 1-alpha
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        #print(val)
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break

    #print(hdv)
    # setup for difference comparison
    hdv.sort()
    diff = (u-l)/20  # differences of 5%

    # if y_i - y_{i-1} > diff then save
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))


    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes

def numeric_univariate(col):
    """ Determine highest desnity intervals. """

    col = col.to_numpy()
    hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(col)


def hdp_example(show_plot=False): 
    """ Plot an example of HDP method on bimodal data.

    Src: https://stackoverflow.com/questions/53671925/highest-density-interval-hdi-for-posterior-distribution-pystan
    """

    # include two modes
    samples = np.random.normal(loc=[-4,4], size=(1000, 2)).flatten()

    # compute high density regions
    hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples)

    plt.figure(figsize=(8,6))

    # raw data
    plt.hist(samples, density=True, bins=29, alpha=0.5)

    # estimated distribution
    plt.plot(x_mu, y_mu)

    # high density intervals
    for (x0, x1) in hpd_mu:
        plt.hlines(y=0, xmin=x0, xmax=x1, linewidth=5)
        plt.axvline(x=x0, color='grey', linestyle='--', linewidth=1)
        plt.axvline(x=x1, color='grey', linestyle='--', linewidth=1)

    # modes
    for xm in modes_mu:
        plt.axvline(x=xm, color='r')

    if show_plot: plt.show()

###################### Decision Tree #############


###################### Run Helpers #############

def run_data_search(df):
    """ Iterate over data columns and perform the following actions...
    1. If type(col) is numeric, run HDP
    2. If type(col) is non-numeric, run DT
    3. Run DT for all interactions

    :param df: (pd.DataFrame) of raw data with correct/incorrect classification
    """

    categoricals  = [x for x in list(df) if 'color' in x or 'gender' in x]

    # univariate loop
    for col_name in df:
        c = df[col_name]

        if col_name in categoricals:
            pass

        else:
            pass




###################
# Run
###################
# Step 1: train our model 
df = read_data()
train_baseline_model(df)

# Step 2: find areas of weakness
hdp_example()
run_data_search(df)


