import pandas as pd
import numpy as np

#############################
# Data Generation/Pulls 
#############################
def make_time_series(n_rows, n_cols):
    """Create a single column of length N.

    src: https://stackoverflow.com/questions/56310849/generate-random-timeseries-data-with-dates

    :param n_rows: (int) 
    :param n_cols: (int) 
    :return: (pd.DataFrame) of generated time series
    """
    
    # create data
    rng = pd.date_range('2000-01-01', freq='d', periods=n_rows)
    df = pd.DataFrame(np.random.rand(n_rows, n_cols), index=rng)

    # "unclean" data
    df = df.apply(lambda x: make_outliers_on_col(x), axis='index')
    df = df.apply(lambda x: make_nan_on_col(x), axis='index')

    return df

def make_outliers_on_col(col, n_per_col=None):
    """Mutate column and create outliers

    :param col: (pd.Series) column to mutate
    :param n_per_col: (int) number of values to mutate
    :return: (pd.Series) mutated series
    """

    if n_per_col is None:
        n_per_col = int(len(col) * 0.01)

    # pad either side of col to allow for interpolation
    idx = np.random.choice(list(range(1, len(col) - 1)), size=n_per_col)
    mutation = 2 + (2 * np.random.rand(n_per_col))
    col.iloc[idx] = col.iloc[idx] + mutation

    return col

def make_nan_on_col(col, n_per_col=None):
    """Mutate column and create outlier

    :param col: (pd.Series) column to mutate
    :param n_per_col: (int) number of values to mutate
    :return: (pd.Series) mutated series
    """

    if n_per_col is None:
        n_per_col = int(len(col) * 0.01)

    idx = np.random.choice(list(range(len(col))), size=n_per_col)
    col.iloc[idx] = np.nan

    return col

#############################
# Visualizations 
#############################
