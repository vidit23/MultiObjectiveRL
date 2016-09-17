import pandas as pd
import Quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = Quandl.get('WIKI/GOOGL')

# Selecting the columns we need
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# New column which measures the change in High & Low
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
# New column which measures the change in stock each day
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col =  'Adj. Close'

# To treat any empty columns as an outlier and you don't want to lose the row
df.fillna(-9999, inplace = True)

# Predict 1% of the database
forecast_out  = int(math.ceil(0.01*len(df)))
print(forecast_out)

# Label for each day becomes some "Adj. Close" of the future
df['label'] = df[forecast_col].shift(-1*forecast_out)
df.dropna(inplace = True)

X = np.array(df.drop(['label'], 1))
Y = np.array(df['label'])

# To make all the feature value between 0 and 1
X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression()
# To use Support Vector Machine
# clf = svm.SVR()

clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)

print(accuracy)
