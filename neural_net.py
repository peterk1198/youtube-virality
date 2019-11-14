
import torch
import torch.nn as nn
# source: 
class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = None #number of features in the featurwe vector
        self.outputSize = 1 #
        self.hiddenSize = None # 2/3 the size of the input lagter ?
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 3 X 2 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    def forward(self, X):
        
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        return self.sigmoid(self.z3) # returns the final activiation function
        
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
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        forw = self.forward(xPredicted)
        print ("Output: \n" + str(forw))
        return forw

    if __name__ == '__main__':
        # for each category name :
        # ---------------TRAINING---------------------
        # get X tensor for the category
        # get y tensor output
        NN = Neural_Network()
        for i in range(1000):
            print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
            NN.train(X, y)
            NN.saveWeights(NN)
        # --------------PREDICT -----------------------------
        # for value in training set xPredicited, expectedOut
        expectedOut = None
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        val = NN.predict()
        if val == expectedOut:
            tp += 1
        elif val != expectedOut and expectedOut == 1:
            fn += 1
        elif val != expectedOut and expectedOut == 0:
            fp += 1
        else: 
            tn +=1 
        #end of loop
        recall = tp / tp + fn
        accuracy = tp + tn / tp + fp + fn + tn
        f1 = 2 * (recall * accuracy) / (recall + accuracy)

