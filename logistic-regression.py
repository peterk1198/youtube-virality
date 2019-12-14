import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# Logistic Regression.
# This program is meant to be used after split-train-test.py. While split-train-test.py also
# runs logistic regression, this file is where we can tune parameters, etc. when the sparse
# matrix has been created and outputted.

# opens the csv of training examples
df = pd.read_csv('./CAvirality.csv')
y = df['viral']

# opens the dataframe that is exported from split-train-test.py
df = pd.read_csv('./df.csv')

# creates scikit learn model.
print('Creating model.')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# fits model.
print('Made Model')
lm = linear_model.LogisticRegression()
print('Fitting Model')
model = lm.fit(X_train, y_train)
print('Predicting Model')
predictions = lm.predict(X_test)

# plots models predictions, etc.
print('Plotting Model')
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
print('Score:', model.score(X_test, y_test))
