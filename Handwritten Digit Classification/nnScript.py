#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import re 
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # Numpy has the exp - 'e' functionality, so convert it to numpy array to 
    # calculate sigmoid.
    sigmoid_function = 1/(1+ np.exp(-z))
    return sigmoid_function

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    #We initialise the matrix for the tranining, validation and testing sets with zeroes and give a shape
    #Similarly we use the test, validation and train label and initialise with 0 
    training_set = np.zeros(shape=(0,784))
    validation_set = np.zeros(shape=(0, 784))
    testing_set = np.zeros(shape=(0, 784))
    train_label = np.zeros(shape=(0,1))
    validation_label = np.zeros(shape=(0,1))
    test_label = np.zeros(shape=(0,1))
    
    #The matrix contains the key as "train0","train1" etc. so we use this key data to define our actual labels
    # We require to split our data into 50000 training examples and 10000 validation examples
    # There are 10 keys, so in each key we give random 1000 eaxmples to validation set, which eventually gives 10000
    # examples to validation set and 50000 to training set
    # If the keyword is "test" we assign it to training set
    
    for i in range(10):
        total_training_set = mat.get('train'+str(i));
        test_set = mat.get('test'+str(i));
        
        index=range(total_training_set.shape[0]);
        train_count=total_training_set.shape[0]
        test_count=test_set.shape[0]
        final_train_count= train_count - 1000
        validation_count=train_count-final_train_count
        for j in range(final_train_count):
            train_label=np.append(train_label,i);
        for j in range(validation_count):
            validation_label=np.append(validation_label,i);   
        for j in range(test_count):
            test_label=np.append(test_label,i); 
            
        permute=np.random.permutation(index)
        temp_train=total_training_set[permute[0:final_train_count],:]
        temp_validation=total_training_set[permute[final_train_count:],:]
  
        training_set=np.vstack([training_set,temp_train]);
        validation_set=np.vstack([validation_set,temp_validation]);
        testing_set=np.vstack([testing_set,test_set]);
        
    
    # Normalization of Data Increased Accuracy by 5%      
    training_set = training_set/255.0
    validation_set = validation_set/255.0
    testing_set = testing_set/255.0  

    # Feature selection
    all_cols = np.concatenate((training_set, validation_set, testing_set), axis=0)

    del_list = np.all(all_cols == all_cols[0,:], axis=0)
    dl = []
    for i in range(len(del_list.tolist())):
        if(del_list[i] == True):
            dl.append(i)
    testing_set = np.delete(testing_set, dl, axis=1)
    validation_set = np.delete(validation_set, dl, axis=1)
    training_set = np.delete(training_set, dl, axis=1)
   
    print('preprocess done')

    return training_set, train_label, validation_set, validation_label, testing_set, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Adding bias term!
    training_data = np.append(training_data, np.ones((training_data.shape[0],1)), 1)  #add column
    # Your code here
    
    # aj = w^t * x
    hiddenLayerOutput = np.dot(training_data, w1.T)
    # zj = Ïƒ(aj) - Sigmoid activation
    hiddenLayerOutput = sigmoid(hiddenLayerOutput)
    # Adding bias term!
    hiddenLayerOutputIncludingBiasTerm = np.append(hiddenLayerOutput,np.ones([hiddenLayerOutput.shape[0],1]),1)  #add column  
    # bl = w^t * zj
    output = np.dot(hiddenLayerOutputIncludingBiasTerm, w2.T)
    # ol = Ïƒ(bl) - Sigmoid 
    output=sigmoid(output)
    
    actualClass = np.zeros((training_data.shape[0],n_class)) 
    
    # Forward Pass complete
    i=0
    for i in range(training_label.shape[0]):
        position = int(training_label[i])
        actualClass[i][position] = 1
    
    # Starting the backward pass
    objectiveValue = 0.0
    # yi * ln(oi)
    intermediate = np.multiply(actualClass,np.log(output))
    # (1 - yi)ln(1 - oi)
    intermediateSecond = np.multiply(np.subtract(1,actualClass),np.log(np.subtract(1,output)))
    # (yi * ln(oi) + (1 - yi)ln(1 - oi))
    intermediateThird = np.add(intermediate,intermediateSecond)
    training_data_size = training_data.shape[0]
    objectiveValue += np.sum(intermediateThird)
    # J(W(1),W(2)) = -1/n *Î£Î£ (yi * ln(oi) + (1 - yi)ln(1 - oi))
    objectiveValue = ((-1)*objectiveValue)/training_data_size
    # Î£Î£ (w1)^2 + (w2)^2
    regularizationIntermediate = (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    # Î»/ 2*n 
    regularizationIntermediateSecond = np.divide(lambdaval,(2*training_data.shape[0]))
    # Î»/ 2*n (Î£Î£ (w1)^2 + (w2)^2)
    regularizationParameter = regularizationIntermediate * regularizationIntermediateSecond
    # Ä´(W(1),W(2)) = J(W(1),W(2)) + Î»/ 2*n (Î£Î£ (w1)^2 + (w2)^2)
    objectiveValueFinal = objectiveValue + regularizationParameter
    # Î´l * zj => (ol- yl)zj
    grad_w2_basic = np.dot(np.transpose(np.subtract(output,actualClass)), hiddenLayerOutputIncludingBiasTerm) 
    # Î£ grad_basic + Î»wj
    grad_w2_regularized = np.add(grad_w2_basic,np.multiply(lambdaval,w2))
    # 1/n(Î£ grad_basic + Î»wj)
    grad_w2 = np.divide(grad_w2_regularized,training_data.shape[0])
    w2_without_bias_node = w2[:,0:w2.shape[1]-1]
    hiddenLayerOutputIncludingBiasTerm = hiddenLayerOutputIncludingBiasTerm[:,0:hiddenLayerOutputIncludingBiasTerm.shape[1]-1]
    grad_w1_basic = np.multiply(np.multiply(np.subtract(1,hiddenLayerOutputIncludingBiasTerm),hiddenLayerOutputIncludingBiasTerm),np.dot(np.subtract(output,actualClass),w2_without_bias_node))
    grad_w1 = np.add(np.dot(grad_w1_basic.T,training_data),np.multiply(lambdaval,w1))/training_data.shape[0]
    obj_val = objectiveValue
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    data = np.insert(data, data.shape[1], 1, axis=1)
    # Ïƒ(w^t * x )
    hidden_layer_output = sigmoid(np.dot(data, np.transpose(w1)))
    hidden_layer_output = np.insert(hidden_layer_output, hidden_layer_output.shape[1], 1, axis=1)
    output_layer_output = sigmoid(np.dot(hidden_layer_output, np.transpose(w2)))
    labels = np.double(np.argmax(output_layer_output, axis=1))
    return labels


    """**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 100}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


# In[ ]:




