import numpy as np
import math
import random

def weight_initialize(X):
    x_shape = X.shape
    W = np.zeros(x_shape[1])
    for j in range(x_shape[1]):
        W[j] = round(random.uniform(-1,1),4)
    b= round(random.uniform(-1,1),4)
    return W,b

def dot_product(X,W):
    result = 0
    for i in range(len(X)):
        result += W[i]*X[i]
    return result

def update_weights(W,b,X,Y):
    for i in range(len(X)):
        W[i] = W[i] + X[i]*Y[0]
    b = b + Y[0]
    return W,b

def perceptron_train(X,Y):
    x_shape = X.shape
    epochs = 10
    a= 0
    W,b = weight_initialize(X)
    for j in range(epochs):
        #print('Epoch: ',j)
        for i in range(x_shape[0]):
            a = dot_product(X[i],W)
            a = a + b
            ay = a*Y[i][0]
            if ay <= 0:
                W,b = update_weights(W,b,X[i],Y[i])
                #print(W,b)
            else:
                pass
    Weights = []
    Weights.append((W,b))
    #print('Final:',W)
    return Weights[0]

def activation(X, W, b):
        act = dot_product(X,W)
        act = act + b
        if act > 0:
          act = 1
        else:
          act = -1            
        return act
    
def perceptron_test(X_test, Y_test, W, b):
        Y_pred = []
        x_shape = X_test.shape
        for i in range(x_shape[0]):
            pred = activation(X_test[i], W, b)
            Y_pred.append(pred)
        count = 0
        for i in range(x_shape[0]):
            if Y_pred[i] == Y_test[i][0]:
               count = count + 1
        #print(count)
        #print(Y_test)
        #print(Y_pred)
        acc = (count/len(Y_pred)) * 100.0
        return acc