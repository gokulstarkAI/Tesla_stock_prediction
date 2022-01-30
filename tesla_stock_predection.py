"""
# Importing libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %matplotlib inline
from sklearn import linear_model

"""# Reading and preparing data"""

df = pd.read_csv('TSLA.csv')
df.head()

df.info()

df.describe()

cdf = df[['High','Low','Open','Volume','Close']]
cdf.head(10)

"""# Linear regression"""

plt.scatter(cdf.High,cdf.Close,color='red')
plt.xlabel('High')
plt.ylabel('Close')
plt.show()

"""# train and test spliting"""

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.High,train.Close,color='red')
plt.xlabel('High')
plt.ylabel('Close')
plt.show()

"""# Model """

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['High','Low','Open','Volume']])
y = np.asanyarray(train[['Close']])
regr.fit(x,y)
print('Coefficients:', regr.coef_)

"""# predict"""

y_hat = regr.predict(test[['High','Low','Open','Volume']])
x = np.asanyarray(test[['High','Low','Open','Volume']])
y = np.asanyarray(test[['Close']])
print('Residual sum of squares: %.2f' %np.mean((y_hat - y)**2))
print('Variance score: %.2f' %regr.score(x,y))

print(y_hat)

"""# Checking how the model works"""

y1 = y.flatten()
y_hat1 = y_hat.flatten()

df2 = pd.DataFrame({'Actual':y1, 'Predicted':y_hat1})
df2.head()

graph = df2.head(20)

graph.plot(kind='bar')

