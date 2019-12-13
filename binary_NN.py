# Template Source : https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
# Date Accessed : November 1st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re
import json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


#-------------------------------------------------NEURAL NETWORK CLASS ------------------------------------------------
# Class definition of a Neural Network
class Neural_Network(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        
        super(Neural_Network,self).__init__()
        # Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        # This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(hiddenSize,outputSize)
        
    def forward(self,x):
        # Applies matrix multiplication beween our weights and our input layer 
        x = self.fc1(x)
        # Activation function is Relu. The Relu fucntion takes the mazimum value between 0 and x
        x = torch.relu(x)
        # Applies our weights to our hidden layer and returns a an output with the same width of our output layer
        x = self.fc2(x)
        return x
        
    # This function takes an input and predicts the class (0 or 1)        
    def predict(self,x):
        # Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        # Picks the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

#--------------------------------------------------- READING DATA ----------------------------------------------------

# intialized dataframes to be empty


dfX = None # dataframe form of input values
dfY = None # dataframe form of output values 
rX =  None # pickle form of dataframe for input features
rY = None #  pickle form of output values 


# open files (input and ooutput)
# check if saved pickle of datframe is availible, if not read the csv file and convert it to a pandas dataframe
try: 
    rX = pd.read_pickle('rX')
    dfX = torch.tensor(rX.values).type(torch.FloatTensor)
except:
    rX = pd.read_csv('df.csv', '\t')
    dfX = torch.tensor(rX.values).type(torch.FloatTensor)
    rX.to_pickle('rX')
try:
    rY = pd.read_pickle('rY')
    dfY = torch.tensor(rY['viral'].values).type(torch.LongTensor)   
    print('Pickle found')
except:
    rY = pd.read_csv('./CAvirality.csv')
    rY.to_pickle('rY')
    dfY = torch.tensor(rY['viral'].values).type(torch.LongTensor)
    print('opened')



# combines the output column with the inputs
combined = rX
feats = rX.columns.values
combined['viral'] = rY['viral']

# splits the dataset into test and train
train, test = train_test_split(combined, test_size=0.2)
yT = train['viral'].values
yT = yT.reshape((-1, 1))
yTrain = torch.tensor(yT)
train.drop(['viral'], axis =  1)

# splits into validation and training sets
test, valid = train_test_split(test, test_size=0.5)

# input features for our training set
xTrain = torch.tensor(train.values)

yT = test['viral'].values
yT = yT.reshape((-1, 1))
test.drop(['viral'], axis =  1)
feats_copy = test.columns.values

# outut values for test set
yTest = torch.tensor(yT)
ylist = sum(yTest.flatten().tolist())

test.drop(['viral'], axis = 1)

# input features for oru test set
xTest = torch.tensor(test.values) 




# -------------------------------------------- RUNNING NEURAL NETWORK ---------------------------------------
#Initialize the model        
model = Neural_Network(xTrain.shape[1], 50 , 2)
#Define loss criterion
criterion = nn.CrossEntropyLoss()

#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Number of epochs
epochs = 1000
#List to store losses
losses = []
for i in range(epochs):
    if i % 10 == 0:
        print(i/epochs)
    #Precit the output for Given input
    y_pred = model.forward(xTrain.float())

    #Compute Cross entropy loss
    loss = criterion(y_pred, torch.max(yTrain, 1)[0])
    #Add loss to the list
    losses.append(loss.item())
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()

# gets the weights of all the words in our feature vetor
mat_w = model.fc1.weight.data.numpy()
# sums up the average weights for each sample
weights = np.sum(mat_w, axis=0)
# orders the list by the words with the highest weight
top_inds = weights.argsort()[-50:][::-1]
curr = 1

# for each of these words we print its rank and the name of the word feature
for i in  top_inds:
    print(str(curr) + ' :' + feats[i])
    curr += 1


# Plots the Loss grah by taking the list of losses over every iteration
plt.plot(list(range(epochs)), losses, linewidth= 3.0)
plt.xlabel("Iterations")
plt.ylabel("Loss (Cross Entropy)")
plt.show()

# makes a prediction given the trainig performed on the model
prediction = model.predict(xTest.float())
for i in range(len(prediction)):
    pred = prediction[i]
    y = yTest[i]
    # checks for False Positive
    if y == 0 and pred == 1:
        x_vec = xTest[i]
        title = ''
        for j in range(len(x_vec)):
            if x_vec[j] == 1:
                title += feats_copy[j] + ' '
    # checks for False Negtive 
    if y == 1 and pred == 0:
        title = ''
        x_vec = xTest[i]
        for j in range(len(x_vec)):
            if x_vec[j] == 1:
                title += feats_copy[j] + ' '


# prints out classifaction information about our input data
print('Accuracy', accuracy_score(prediction, yTest))
print('F1_score', accuracy_score(prediction,yTest))
print(classification_report(model.predict(xTest.float()),yTest))
print(confusion_matrix(model.predict(xTest.float()),yTest))

