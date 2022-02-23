# Desc: OLS from scratch code for post 36-37
# Author: Michael Berk
# Date: Winter 2022

import numpy as np
import pandas as pd
import plotly.express as px

from numpy import matmul
from numpy.linalg import inv
import statsmodels.api as sm

np.random.seed(99)
show = True 

# Step 0: Create a dataset
const = np.repeat(1, 100)
age = 10*np.array([np.random.uniform(size=100)]).flatten()
bamboo = 5*np.array([np.random.uniform(size=100)]).flatten()
has_siblings = np.where(np.random.uniform(size=100) > 0.50, 1, 0)

baby_panda_weight = (np.random.normal(size=100) + age + bamboo + has_siblings).flatten()

if not show:
    plot_df = pd.DataFrame(dict(age=age, baby_panda_weight=baby_panda_weight))
    px.scatter(plot_df, x='age', y='baby_panda_weight',
               template='plotly_white',
               title='Raw Data of Weight vs. Age').update_traces(marker=dict(color='#B80C09')).show()

X = np.array([const, age, bamboo, has_siblings]).T
Y = np.array(baby_panda_weight)
beta = np.array([1,2,3,4]) # dummy beta values

assert X.shape[0] == 100 and X.shape[1] == 4 
assert Y.shape[0] == 100

# Step 1: calculate residauls
def calculate_resid(x, y, beta):
    resid = y - np.sum(matmul(x, beta))
    return resid 

resid = calculate_resid(X, Y, beta)
if show: print(f'Resid: {resid}')

# Setp 2: calculate ssq
def calculate_ssq(resid):
    return np.sum(matmul(resid.T, resid))

ssq = calculate_ssq(resid)
if show: print(f'SSQ: {ssq}')

# Step 3: calcualte derivative
def derivative(x, y, beta):
    term_1 = matmul(-2 * x.T, y)
    term_2 = matmul(matmul(2 * x.T, x), beta)
    return term_1 + term_2

der = derivative(X, Y, beta)
if show: print(der)

# Setp 4: solve for the best beta
def best_beta(x, y):
    a = matmul(x.T, x)
    b = inv(a)
    c = matmul(b, x.T)
    d = matmul(c, y)

    return d

best_betas = best_beta(X, Y)
if show: print(best_betas)

# Step 5: check against statsmodels
m = sm.OLS(Y, X).fit()
print(m.summary())

assert all(np.round(best_beta(X, Y), 4) == np.round(m.params, 4))



    


