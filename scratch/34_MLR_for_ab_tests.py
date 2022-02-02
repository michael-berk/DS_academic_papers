# Description: potential implementation for IBM's FreaAI method
# Author: Michael Berk
# Date: Winter 2022

import numpy as np
import pandas as pd 
import plotly.express as px

import statsmodels.api as sm
from scipy.stats import ttest_ind
np.random.seed(1)

############ Setup #############
def create_experiment(mu_c = 5, percent_lift=0.01, scale=1, n=10000, seasonal=False):
    """
    Create and return two experiment data sets. Each value
    corresponds to the mean for an experimental unit.
    """

    control = np.random.normal(loc=mu_c, scale=scale, size=n)
    treat = np.random.normal(loc=mu_c + mu_c*percent_lift, scale=scale, size=n)
    covariate = np.random.chisquare(5, size=n)

    if seasonal:
        s = np.sin(np.arange(len(control))) + np.array([1 if np.random.uniform(len(control)) > 0.7 else 0])
        return (treat * s + covariate, control * s + covariate, covariate, s)

    return (treat, control, covariate)

e = create_experiment(seasonal=True)

# plot
#plot_df = pd.DataFrame(dict(t=e[0], c=e[1], x=list(range(len(e[0])))))
#px.scatter(plot_df, x='x', y=['t','c']).show()

######### Lift Calcs #########
def lift(t, c, covariate, s=None):
    # calculate lifts
    mu_t, mu_c = np.mean(t), np.mean(c)
    ATE = np.mean(mu_t - mu_c)
    percent_lift = ATE / np.mean(mu_c)

    # t-test
    p = ttest_ind(t, c).pvalue

    return ATE, percent_lift, p

print(lift(*e))

def lm_lift(t, c, covariate, s=None):

    is_treat = np.append(np.repeat(1, len(t)), np.repeat(0, len(c)))

    if s is not None:
        #x = np.array([np.repeat(1, len(t) + len(c)), is_treat, np.append(s, s), np.append(covariate, covariate)]).T
        x = np.array([np.repeat(1, len(t) + len(c)), is_treat, np.append(s, s)]).T
    else:
        x = np.array([np.repeat(1, len(t) + len(c)), is_treat, np.append(covariate, covariate)]).T

    lm = sm.OLS(np.append(t, c), x).fit()
    print(lm.summary())
    return lm.params[0], lm.pvalues[0]

print(lm_lift(*e))




