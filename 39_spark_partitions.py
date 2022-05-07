from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

import os
import inspect

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

############# Setup ##############
blue = '#177BCD'
light_blue = '#9ED8DB'
yellow = '#EAB464'
red = '#B80C09'
black = '#040F16'
gray = '#2C514C'
light_gray = '#9A9384'
green = '#57A773'

show = False
spark = SparkSession.builder.getOrCreate()

def create_df():
    a = list(range(1000))
    b = [str(x) for x in a]
    c = np.random.uniform(size = len(a))

    return spark.createDataFrame(pd.DataFrame(dict(a=a,b=b, c=c)))

df = create_df()
n_cores = os.cpu_count()
print(f'Machine with: {n_cores}')

######### Example of Skew #########
even = [100 for i in range(n_cores)]
uneven = [int(np.random.uniform() * 200) for i in range(n_cores)]

def create_hist(y, color, title):
    fig = go.Figure(data=go.Bar(x=[f'Core {i}' for i in range(len(y))], y=y))

    fig.update_layout(
        title_text = title,
        template = 'plotly_white'
    )
    fig.update_traces(marker_color = color)
    fig.update_yaxes(title_text = 'N Rows of Data')

    return fig

if show:
    create_hist(even, blue, 'No Data Skew (Even Data Distribution)').show()
    create_hist(uneven, red, 'Has Data Skew (Even Data Distribution)').show()

########### Determine Skew ##########
if show:
    # method 1
    df.groupBy(F.spark_partition_id()).count().show()

    # method 2
    glomed = df.rdd.glom().collect()
    for g in glomed:
        print(g[:2])

########## Correct Skew ########
df_part = df.repartition(8, 'a')
df_part.groupBy(F.spark_partition_id()).count().show()

df_part = df.withColumn('salt', F.rand())
df_part.repartition(8, 'salt')
df_part.groupBy(F.spark_partition_id()).count().show()

