#Source: https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c

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
#our class must extend nn.Module

class Neural_Network(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        
        super(Neural_Network,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(hiddenSize,outputSize)
        
    #This must be implemented
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. Feel free to experiment with this
        x = torch.tanh(x)
        #This produces output
        x = self.fc2(x)
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

dfX = None
dfY = None
rX =  None
rY = None

# open files and read in data
try: 
    rX = pd.read_pickle('rX')
    dfX = torch.tensor(rX.values).type(torch.FloatTensor)
    print('Pickle found')
except:
    print('opening')
    rX = pd.read_csv('df.csv', '\t')
    dfX = torch.tensor(rX.values).type(torch.FloatTensor)
    rX.to_pickle('rX')
    print('opened')
try:
    rY = pd.read_pickle('rY')
    dfY = torch.tensor(rY['viral'].values).type(torch.LongTensor)   
    print('Pickle found')
except:
    rY = pd.read_csv('./CAvirality.csv')
    rY.to_pickle('rY')
    dfY = torch.tensor(rY['viral'].values).type(torch.LongTensor)
    print('opened')

# combine with y column

combined = rX
combined['viral'] = rY['viral']
train, test = train_test_split(combined, test_size=0.2)
yT = train['viral'].values
yT = yT.reshape((-1, 1))
yTrain = torch.tensor(yT)
train.drop(['viral'], axis = 1)
xTrain = torch.tensor(train.values)


yT = test['viral'].values
yT = yT.reshape((-1, 1))
yTest = torch.tensor(yT)
test.drop(['viral'], axis = 1)
xTest = torch.tensor(test.values) 

print('Creationg Neural_Network')
#Initialize the model        
model = Neural_Network(xTrain.shape[1], 100, 1)
#Define loss criterion
criterion = nn.CrossEntropyLoss()
#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Number of epochs
epochs = 1000
#List to store losses
losses = []
for i in range(epochs):
    #Precit the output for Given input
    y_pred = model.forward(xTrain.float())
    #Compute Cross entropy loss
    loss = criterion(y_pred, yTrain.squeeze(1))
    #Add loss to the list
    losses.append(loss.item())
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()
