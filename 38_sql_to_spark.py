# Desc: examples of sql and its pyspark equivalents
# Date: Winter 2022
# Author: Michael Berk

import numpy as np 
import pandas as pd

import pyspark
from pypspark.sql import functions as F
from pypspark.sql import types as T

spark = pyspark.SparkSession()
# TODO get query plan func call

########### Section 0 #########
########### Create Data #########
# create base data
n = 100
a = [i for i in range(n)]
b = 50 > np.array(a)
c = np.sin(np.array(a))
d = np.repeat('hi', size=n)

df = pd.DataFrame(dict(a=a, b=b, c=c, d=d))

########### Section 1 #########
########### SELECT #########
'''
SELECT a > 50 = b
FROM df
'''
sql = 'a > 50 = b' # SELECT a > 50 = b
df.selectExpr(sql)
df.select(F.expr(sql))
df.select(F.col(sql))

# Alias
out1 = df.selectExpr(sql + 'AS my_col')
out2 = df.select(F.col(sql).alias('AS my_col'))
assert out1 == out2

########### FROM #########
'''
WITH my_cte AS (
    SELECT a, c
    FROM df
)

SELECT a * c AS ac
FROM my_cte
'''
my_cte = df.selectExpr('a, c')
out_1 = my_cte.selectExpr('a * c AS ac')

out_2 = (df.selectExpr('a, c')
        .selectExpr('a * c AS ac'))

########### WHERE #########
'''
SELECT a
WHERE b = True
    AND c > 0
    AND d  = 'hi'
'''

df.filter(F.expr('b') == True).filter(F.expr('c') > 0).filter(F.'d' == 'hi')

df.where("b = True AND c > 0 AND d = 'hi'")
df.filter("b = True AND c > 0 AND d = 'hi'")

# Q: how to compare spark type col with python type (bool)
# Q: difference bwetween filter and where


########### Section 2 #########
########### UNNION #########
'''
SELECT a
FROM df

UNION ALL

SELECT c
FROM df
'''

df.selectExpr('a').union(df.selectExpr('c')
########### JOIN #########

########### GROUP BY #########

########### ORDER BY #########
'''
SELECT a, b
FROM df
ORDER BY a DESC, b ASC
'''
out_1 = df.selectExpr('a,b').orderBy('a').desc().orderBy('b').asc()
out_2 = df.select('a,b').orderBy(F.col('a').desc(), F.col('b').asc())
# CAN DO  asc_nulls_first and such

#Q : can you just chain WHERE/ORDER BY and get the same result?


########### LIMIT #########
'''
SELECT a
LIMIT 5
'''
df.select('a').limit(5)

########### HAVING? #########
# Q: COALESCE?
