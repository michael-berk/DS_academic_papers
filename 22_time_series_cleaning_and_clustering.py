# Description: time series clustering and cleaning based on paper21
# Post: https://towardsdatascience.com/how-to-improve-deep-learning-forecasts-for-time-series-part-2-c11efc8dfee2
# Author: Michael Berk
# Date created: 2021-10-22

import helpers.helpers as h

import numpy as np
import random
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.holtwinters import Holt
from scipy.cluster.vq import kmeans2, whiten

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# create data and show
df = h.make_time_series(n_rows=1000, n_cols=10)

raw_data_plot = px.line(df, template='none', title='Uncleaned Raw Data')
raw_data_plot.update_layout(xaxis_title="Date", yaxis_title="Gold Price", legend_title="Individual Gold Market")
raw_data_plot.show()

#########################
# Data cleaning
#########################
# remove outliers
def nullify_outliers(col):
    """ Remove outliers using seasonal decomposition

    :param col: (pd.Series) df to remove outliers for
    :return: (pd.Series) df with linearly interpolated values
    """

    decomp = seasonal_decompose(col, model='additive')
    resid = np.array(decomp.resid)

    # na fill outliers
    per_25, per_75 = np.percentile(resid[~np.isnan(resid)], [25, 75])
    upper = per_75 + (per_75 - per_25) * 1.5
    lower = per_25 - (per_75 - per_25) * 1.5
    col.iloc[np.argwhere((resid > upper) | (resid < lower)).flatten()] = np.nan

    return col

# iterpolate nulls
# src: https://stackoverflow.com/questions/31332981/pandas-interpolation-replacing-nans-after-the-last-data-point-but-not-before-th/38325187
df = df.interpolate(method='spline', order=1, limit=10, limit_direction='both')

# interpolate outliers
df = df.apply(lambda x: nullify_outliers(x), axis='index')
df = df.interpolate(method='spline', order=1, limit=10, limit_direction='both')

# plot
clean_data_plot = px.line(df, template='none', title='Cleaned Raw Data')
clean_data_plot.update_layout(xaxis_title="Date", yaxis_title="Gold Price", legend_title="Individual Gold Market")
clean_data_plot.show()

#########################
# Distance-Based Clustering 
#########################
# src 0: https://towardsdatascience.com/time-series-hierarchical-clustering-using-dynamic-time-warping-in-python-c8c9edf2fda5
# src 1: https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3
# src 2: http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html

#########################
# Feature-Based Clustering 
#########################
def extract_features(df, feature_set):
    """ Extract TS-specific features from the DS

    :param df: (pd.DataFrame)
    :param feature_set: (int) where...
        1 corresponds to ts-specific features.
        2 corresponds to signal-specific features.
    :return: (np.array) feature set
    """

    # TS-specific features
    if feature_set == 1:
        autocorrelation = df.apply(lambda x: acf(x, nlags=3), axis='index')
        partial_autocorrelation = df.apply(lambda x: pacf(x, nlags=3), axis='index')

        return (np.array(autocorrelation).astype(float), np.array(partial_autocorrelation).astype(float))


    # Signal-processing-specific features
    elif feature_set == 2:
        fast_fourier_transform = df.apply(lambda x: np.fft.fft(x), axis='index')
        variance = df.apply(lambda x: np.var(x), axis='index')

        return (fast_fourier_transform.astype(float), np.array(variance).astype(float))

# get ts-specific (1) and signal-specific feature sets (2)
feature_set_1 = extract_features(df, feature_set=1)
feature_set_2 = extract_features(df, feature_set=2)

# whiten features (divide by stdev) and average over rows if needed
w_1 = np.array([np.mean(whiten(x), axis=0) for x in feature_set_1])
w_2 = np.array([whiten(x) for x in feature_set_2])
w_2[0] = np.mean(w_2[0], axis=0).flatten()

# setup loop
titles = ('Clustering Using Time Series Features', 'Clustering Using Signal-Porcessing Features')
features = [w_1.T, np.array([w_2[0], w_2[1]]).T]
axis_titles = (('Autocorrelation','Partial Autocorrelation'), ('Fast Fourier Transform','Varaince'))
cluster_1s = []
cluster_2s = []

# run kmeans
for t, a, f in zip(titles, axis_titles, features):
    # cluster
    out = kmeans2(f, 2)
    cluster_centers, labs = out 
    #print(out)

    # display 
    plot_df = pd.DataFrame(cluster_centers)

    fig = px.scatter(template='plotly_white', title=t)
    fig.add_trace(go.Scatter(x=plot_df.iloc[:,0], y=plot_df.iloc[:,1], 
                             marker=dict(color=['red', 'green']), 
                             mode='markers',
                             marker_symbol=3))
    fig.add_trace(go.Scatter(x=f.T[0], y=f.T[1], 
                             marker=dict(color=[['red', 'green'][int(l)] for l in labs]), 
                             mode='markers',
                             marker_symbol=5))
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(xaxis_title=a[1], yaxis_title=a[0])
    fig.show()

    # save for modeling
    cluster_1s.append(df.loc[:, np.array(labs).astype(bool)])
    cluster_2s.append(df.loc[:, ~np.array(labs).astype(bool)])

#########################
# Vanilla LSTM
#########################
# src: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

def fit_and_display_model(c, look_back=1, n_epoch=10):
    """ Restucture data, fit LSTM, and display results

    :param c: (pd.DataFrame) data that corresponds to a given cluster
    :param look_back: (int) n shifts between x and y - this is the AR param
    :param n_epoch: (int) 
    """
    ############### Pepare data ################
    # choose cluster for fitting and average
    c = pd.DataFrame(np.mean(c, axis='columns'))

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(c)

    # create train test split
    cutoff = int(len(c.index) * 0.7)
    train, test = c.iloc[:cutoff], c.iloc[cutoff:]

    # shift the data by 1 (to create autoregressive nature)
    trainX, trainY  = np.array(train)[:-look_back], np.array(train.shift(look_back))[look_back:]
    testX, testY = np.array(test)[:-look_back], np.array(test.shift(look_back))[look_back:]

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    ############### Model ################
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=n_epoch, batch_size=1, verbose=2)

    # eval
    trainPredict = pd.DataFrame(model.predict(trainX))
    testPredict = pd.DataFrame(model.predict(testX))

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    ############### Display Output ################
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
    print('Train Score: %.5f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
    print('Test Score: %.5f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back):len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

# test each cluster
cs = [cluster_1s[0], cluster_2s[0], cluster_1s[1], cluster_2s[1]] 
for i, c in enumerate(cs):
    print(f"{axis_titles[i % 2]} cluster {(i % 2) + 1}")
    fit_and_display_model(c, look_back=1, n_epoch=10)


