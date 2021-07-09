import numpy as np
import math
import random

def Calculate_distance(x1, x2):
    length = len(x1)
    #print(length)
    dist = 0
    for i in range(length):
        dist += pow((x1[i] - x2[i]), 2)
    dist = math.sqrt(dist)
    return dist

def sorted_neighbor(dist_from_train,k):
    index_arr = list(range(len(dist_from_train)))
    #print(index_arr)                 
    for i in range (1,len(dist_from_train)):
        val = dist_from_train[i][1]
        dummy = dist_from_train[i]
        tracker = index_arr[i]
        j = i-1
        while j >=0 and val < dist_from_train[j][1]:
            dist_from_train[j+1] = dist_from_train[j]
            index_arr[j+1] = index_arr[j]
            j = j-1
        dist_from_train[j+1]=dummy
        index_arr[j+1]=tracker
        
    #print(index_arr)    
    #print(dist_from_train)
    nn = []
    track_index=[]
    for i in range(k):
        nn.append(dist_from_train[i][0])
        track_index.append(index_arr[i])
    return nn,track_index

def nearest_neighbor(X_train, X_test, k):
    dist_from_train = []
    length = len(X_test)
    for i in range(len(X_train)):
        dist = Calculate_distance(X_test, X_train[i])
        dist_from_train.append((X_train[i], dist))
    #print(dist_from_train)
    nn,track_index = sorted_neighbor(dist_from_train,k)
    return nn,track_index

def sign_pred(track,Y_train):
    unique_count = 0
    sign=[]
    for i in Y_train:
        if i not in sign:
            sign.append(i)
            unique_count= unique_count +1
    count_arr= [0]*unique_count
    for i in track:
        for j in range(len(sign)):
            if sign[j] == Y_train[i][0]:
                #print(sign[j],Y_train[i][0])
                count_arr[j] = count_arr[j]+1
    max_pred=0
    max_index=0
    for i in range(len(count_arr)):
        if max_pred < count_arr[i]:
            max_pred = count_arr[i]
            max_index = i
    #print(max_pred)
    return sign[max_index][0]

def Accuracy(pred, Y_test):
    correct_pred = 0
    for x in range(len(Y_test)):
        #print(Y_test[x][0])
        if Y_test[x][0] == pred[x]:
            correct_pred = correct_pred + 1
    acc = (correct_pred/len(Y_test)) * 100.0
    return acc

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    preds=[]
    for x in range(len(X_test)):
        #print(X_test[x])
        neighbors,track = nearest_neighbor(X_train, X_test[x], K)
        rs = sign_pred(track,Y_train)
        preds.append(rs)
        #print(preds)
        #print('> predicted=' + str(rs) + ', actual=' + str(Y_test[x][0]))
    accuracy = Accuracy(preds, Y_test)
    #print(accuracy)
    #print('Accuracy: ' + str(accuracy) + '%')
    return accuracy

def choose_K(X_train,Y_train,X_val,Y_val):
    best_acc = 0
    K = len(X_train)+1
    for k in range(1,K):
        if k % 2 != 0 :
            #print('FOR K = ',k)
            acc = KNN_test(X_train,Y_train,X_val,Y_val,k)
            if best_acc < acc:
                best_acc = acc
                best_k=k
    return best_k