import pandas as pd
import re
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# split data set in 80/10/10

df = pd.read_csv('./CAvirality.csv') # load the dataset as a pandas data frame

data = df[['title', 'category_name']]
y = df['viral']

corpus = defaultdict(int)

for index, row in data.iterrows():
	# Remove all non alpha-numeric characters and split them by a space.
	T = re.sub(r'\W+', ' ', row['title'])
	l = T.split(' ')
	for word in l:
		corpus[word.lower()] += 1

with open('corpus.json', 'w') as fp:
	json.dump(corpus, fp, indent=4)

print('######### Json Dumped #########')

temp_d = defaultdict(list)

for index, row in tqdm(data.iterrows()):
	sparse = np.zeros(len(corpus))
	T = re.sub(r'\W+', ' ', row['title'])
	l = T.split(' ')

	for i, word in enumerate(corpus):
		if word in l:
			sparse[i] = 1

	temp_d['title'].append(tuple(sparse))
	temp_d['category_name'].append(row['category_name'])

print('created temp_d')

df = pd.DataFrame.from_dict(temp_d)
# df.to_csv('df.csv', sep='\t')

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

lm = linear_model.LogisticRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
print('Score:', model.score(X_test, y_test))








	