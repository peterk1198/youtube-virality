import pandas as pd
import re
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from random import *

# split data set in 80/10/10

df = pd.read_csv('./CAvirality.csv') # load the dataset as a pandas data frame

data = df[['title', 'category_name']]
y = df['viral']
print('Exporting y to csv.')
y.to_csv('y.csv', sep='\t')

corpus = defaultdict(int)
categories = defaultdict(int)

for index, row in data.iterrows():
	# Remove all non alpha-numeric characters and split them by a space.
	T = re.sub(r'\W+', ' ', row['title'])
	l = T.split(' ')
	categories[row['category_name']] += 1
	for word in l:
		corpus[word.lower()] += 1

with open('corpus.json', 'w') as fp:
	json.dump(corpus, fp, indent=4)

print('######### Json Dumped #########')

lattice = []

for index, row in tqdm(data.iterrows()):
	temp_d = {}
	T = re.sub(r'\W+', ' ', row['title'])
	l = T.split(' ')

	for cat in categories:
		if cat == row['category_name']:
			temp_d[cat] = 1
		else:
			temp_d[cat] = 0

	for word in corpus:
		if word in l:
			temp_d[word] = 1
		else:
			temp_d[word] = 0
	lattice.append(temp_d)

# print('Exporting temp_d to json.')
# with open('temp_d.json', 'w') as fp:
# 	json.dump(lattice, fp, indent=4)

df = pd.DataFrame(lattice)

# print('Exporting df to csv.')
# df.to_csv('df.csv', sep='\t')

print('Creating model.')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

print('Made Model')
lm = linear_model.LogisticRegression()
print('Fitting Model')
model = lm.fit(X_train, y_train)
print('Predicting Model')
predictions = lm.predict(X_test)

#print('Plotting Model')

print('Score:', metrics.accuracy_score(y_test, predictions))
print('f1:', metrics.f1_score(y_test, predictions))
print("accuracy", metrics.precision_score(y_test, predictions))
print("recall", metrics.recall_score(y_test, predictions))
dummy = []
for i in range(len(predictions)):
	dummy.append(randint(0,1))
print('d Score:', metrics.accuracy_score(y_test, dummy))
print('d f1:', metrics.f1_score(y_test, dummy))
print("dummy accuracy", metrics.precision_score(y_test, dummy))
print("d recall", metrics.recall_score(y_test, dummy))

print("actual", confusion_matrix(y_test, predictions))
print("dummy", confusion_matrix(y_test, dummy))



	