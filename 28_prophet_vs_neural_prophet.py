# Description: comparison between FB Prophet and NeuralProphet
# Author: Michael Berk
# Date created: 2022-12-08

import numpy as np
import pandas as pd

###########################
# Data
###########################
def read_data():
    """ 
    Read data into pandas df. Open raw file for header descriptions.
    Source: https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48
    """

    return pd.read_excel('EnergyTSData/Region_CAL.xlsx', sheet_name='Published Daily Data', header=0, engine="openpyxl")


###########################
# Prophet (v1) 
###########################

###########################
# NeuralProphet (v2) 
###########################
