import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Logistic Regression

df = pd.read_csv('./CAvirality.csv')
y = df['viral']

temp_d = {}
with open('temp_d.json') as json_file:
	temp_d = json.load(json_file)

print('Read Json, converting to DF..')
df = pd.DataFrame.from_dict(temp_d)

print('Creating model.')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

print('Made Model')
lm = linear_model.LogisticRegression()
print('Fitting Model')
model = lm.fit(X_train, y_train)
print('Predicting Model')
predictions = lm.predict(X_test)

print('Plotting Model')
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
print('Score:', model.score(X_test, y_test))
