import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# source: 
class Neural_Network(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = inputSize #number of features in the featurwe vector
        self.outputSize = outputSize #
        self.hiddenSize = hiddenSize # 2/3 the size of the input lagter ?
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # n x ]
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o
    
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self, xPredicted):
        pred = F.softmax(self.forward(xPredicted))
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(pred))
        return pred


# for each category name :
# ---------------TRAINING---------------------

dfX = None
dfY = None
rX =  None
rY = None


try: 
    rX = pd.read_pickle('rX')
    dfX = torch.tensor(rX.values.astype(np.float32))
    print('Pickle found')
except:
    print('opening')
    rX = pd.read_csv('df.csv', '\t')
    dfX = torch.tensor(rX.values.astype(np.float32))
    rX.to_pickle('rX')
    print('opened')

try:
    rY = pd.read_pickle('rY')
    dfY = torch.tensor(rY['viral'].values)   
    print('Pickle found')
except:
    rY = pd.read_csv('./CAvirality.csv')
    rY.to_pickle('rY')
    dfY = torch.tensor(rY['viral'].values)
    print('opened')

# combine with y column

combined = rX
combined['viral'] = rY['viral']
combined = combined[:40]
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

print('---------------------------------RUNNING NUERAL NET------------------------------')

# creating tensor from targets_df 
NN = Neural_Network(xTrain.shape[1], 3, 1)
for i in range(1000):
    print('iteration', str(i))
    print ("#" + str(i) + " Loss: " + str(torch.mean((yTrain - NN(xTrain.float()))**2).detach().item()))  # mean sum squared loss
    NN.train(xTrain.float(), yTrain)
    NN.saveWeights(NN)
# --------------PREDICT -----------------------------
# for value in training set xPredicited, expectedOut

pred = NN.predict(xTest.float()).numpy()
obse = yTest.numpy()
guess = [item for sublist in pred for item in sublist]
actual = [item for sublist in obse for item in sublist]
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(guess, actual)
print('F1 score: %f' % f1)



