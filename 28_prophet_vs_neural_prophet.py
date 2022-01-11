# Description: comparison between FB Prophet and NeuralProphet
# Post: https://towardsdatascience.com/prophet-vs-neuralprophet-fc717ab7a9d8
# Author: Michael Berk
# Date created: 2022-12-08

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric
from prophet.diagnostics import performance_metrics, cross_validation

from neuralprophet import NeuralProphet, set_log_level

###########################
# Data + helpers
###########################
def read_data():
    """ 
    Read data into pandas df. Open raw file for header descriptions.
    Source: https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48
    """

    return pd.read_excel('EnergyTSData/Region_CAL.xlsx', sheet_name='Published Daily Data', header=0, engine="openpyxl")

def read_and_clean_df():
    """
    Convert data to a univariate ts with ds and y col.
    """

    df = read_data()
    df = df[['Local date','D']]
    df.columns = ['ds','y']

    return df.iloc[:-2,:]

def plot_data():
    """
    Read clean data and plot. 

    """

    df = read_and_clean_df()
    fig = px.scatter(df, x='ds', y='y', title='California Energy Demand', template='simple_white')
    fig.show()

def pull_gov_preds_and_get_accuracy():
    """
    Pull the day-ahead government forecasts.

    """
    df = read_data()

    return (accuracy(df['DF'].shift(1), df['D']))

def accuracy(obs, pred):
    """
    Calculate accuracy measures

    :param obs: pd.Series of observed values
    :param pred: pd.Series of forecasted values
    :return: dict with accuracy measures
    """

    obs, pred = np.array(obs.dropna()), np.array(pred.dropna())

    assert len(obs) == len(pred), f'accuracy(): obs len is {len(obs)} but preds len is {len(pred)}'

    rmse = np.sqrt(np.mean((obs - pred)**2))
    mape = np.mean(np.abs((obs - pred) / obs)) 

    return (rmse, mape)

###########################
# Prophet (v1) 
###########################
# base code from docs on Prophet
def fit_prophet(df):
    """
    Fit Prophet model and return 
    Source: https://facebook.github.io/prophet/docs/quick_start.html#python-api
    """

    # fit model
    m = Prophet()
    m.fit(df)

    # create forecast
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # create plots
    pred_plot = plot_plotly(m, forecast)
    comp_plot = plot_components_plotly(m, forecast)

    return (m, forecast, pred_plot, comp_plot)

def eval_prophet(m):
    """
    Perform CV on data and evaluate. Note TS CV differs from regular CV.
    IMPORTANT: this is correct eval method according to the docs, but differs from NeuralProphet 
               so I built a custom function.

    """

    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')

    return (df_cv, performance_metrics(df_cv))


###########################
# NeuralProphet (v2) 
###########################
# base code from docs on NeuralProphet
def fit_neural(df, params=None):
    """
    FIt NeuralProphet and return key objects.
    """

    # fit 
    m = NeuralProphet(**params) if params is not None else NeuralProphet()
    metrics = m.fit(df, freq="D")

    df_future = m.make_future_dataframe(df, periods=365)
    forecast = m.predict(df_future, raw=True, decompose=False) if params is not None else m.predict(df_future)

    if params is None:
        fig_forecast = m.plot(forecast)
        fig_components = m.plot_components(forecast)
        fig_params = m.plot_parameters()

        return (m, forecast, fig_forecast, fig_components, fig_params) 

    else:
        return (None, forecast, None, None)

def eval_neural(df):
    """
    Perform cross validation on our model.
    IMPORTANT: this is correct eval method according to the docs, but differs from Prophet 
               so I built a custom function.
    """

    # setup k fold CV
    METRICS = ['SmoothL1Loss', 'MAE', 'RMSE']

    folds = NeuralProphet().crossvalidation_split_df(df, freq="D", k=5, fold_pct=0.20, fold_overlap_pct=0.5)

    # return dfs
    metrics_train = pd.DataFrame(columns=METRICS)
    metrics_test = pd.DataFrame(columns=METRICS)

    # CV
    for df_train, df_test in folds:
        m = NeuralProphet()
        train = m.fit(df=df_train, freq="D")
        test = m.test(df=df_test)
        metrics_train = metrics_train.append(train[METRICS].iloc[-1])
        metrics_test = metrics_test.append(test[METRICS].iloc[-1])

    return (metrics_train, metrics_test)

###########################
# All in One Function
###########################
def cv_run_both_models(df, neural_params):
    """
    Create a CV dataset and run both models and return accuracies. Note that both models
    have their own eval methods, but they differ so I built a custom func that does the 
    same thing. 

    :param df: pd.DataFrame of Prophet-specified format
    :param neural_params: dict of params to be passed to NeuralProphet
    :return: tuple of training accuracies
    """

    # create train test splits (test = 365 days, train = all prior data, increment = 180 days)
    train_test_split_indices = list(range(365*2, len(df.index) - 365, 180))
    train_test_splits = [(df.iloc[:i, :], df.iloc[i:(i+365), :]) for i in train_test_split_indices]

    rmse_p, mape_p = [], []
    rmse_n, mape_n = [], []
    n_training_days = []

    # loop through train/test splits
    for x in train_test_splits:
        train, test = x
        n_training_days.append(len(train.index))

        # train Prophet and get accuracy 
        _, forecast, *_ = fit_prophet(train)
        rmse, mape = accuracy(test['y'], forecast.loc[test['y'].index, 'yhat'])
        rmse_p.append(rmse)
        mape_p.append(mape)

        # train NeuralProphet and get accuracy 
        _, forecast, *_ = fit_neural(train, neural_params)
        rmse, mape = accuracy(test['y'], pd.Series(np.array(forecast.iloc[:, 1:]).flatten()))
        rmse_n.append(rmse)
        mape_n.append(mape)

    return pd.DataFrame(dict(
        n_training_days=n_training_days,
        prophet_RMSE=rmse_p,
        neural_RMSE=rmse_n,
        prophet_MAPE=mape_p,
        neural_MAPE=mape_n
    ))



###########################
# Run
###########################
# Read data
df = read_and_clean_df()
'''
plot_data()  
'''

########## Prophet ###############
# fit model and store output
'''
m1, forecast1, forecast_plot1, component_plot1 = fit_prophet(df)
cv_df, prophet_accuracy_df = eval_prophet(m1)

# show plots
forecast_plot1.show()
component_plot1.show()
print(prophet_accuracy_df.describe())
'''

########## Neural Prophet ##########
# fit model and store output
'''
m2, forecast2, forecast_plot2, component_plot2, params_plot2 = fit_neural(df)
metrics_train, metrics_test = eval_neural(df)

print(metrics_train.describe())
print(metrics_test.describe())

plt.show()
'''

############# Both Models' Training Accuracy ##########
# set model parameters to use AR-Net
"""
neural_params =  dict(
    n_forecasts=365,
    n_lags=30,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    batch_size=64,
    epochs=200,
    learning_rate=0.03
)

out = cv_run_both_models(df, neural_params)
print(out)
"""

############### Comparison with gov ##############
'''
print(pull_gov_preds_and_get_accuracy())
'''
