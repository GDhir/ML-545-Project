import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def read_data():
    """
    xy0, xy1, xy2 represent files corresponding to three outputs (CL, CM and CD)
    xy0:
    x: (x_geo (7 thickness, 7 cmaber) Ma, alpha),
    y: (CL), CM, CD,
    pCL/px, pCM/px, pCD/px
    """
    xy0 = np.loadtxt('M0CFDdata.txt')
    xy1 = np.loadtxt('M1CFDdata.txt')
    xy2 = np.loadtxt('M2CFDdata.txt')
    xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))
    dim = 16
    x = xy[:, :dim]
    y = xy[:, dim:dim+3]
    scaler = MinMaxScaler()
    N_train = np.int(len(y[:,0])*0.8) #Number of train data
    x_train = scaler.fit_transform(x[:N_train,:])
    x_test = scaler.transform(x[N_train:,:])
    y_train = y[:N_train,:]
    y_test = y[N_train:,:]
    y_train_n =scaler.fit_transform(y_train)
    y_test_n = scaler.transform(y_test)
    return x_train, y_train_n, x_test, y_test_n , y_train, y_test

def denormalize(df,norm_data):
    df = df.reshape(-1,1)
    scl = MinMaxScaler()
    scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new

def hidden_layer(prev_layer,n_neur):
    W = tf.Variable(tf.random_uniform([n_neur,n_neur]))
    b = tf.Variable(tf.zeros([n_neur]))
    layer = tf.add(tf.matmul(prev_layer,W), b)
    layer = tf.nn.relu(layer) #activation function
    return W, b, layer

def neural_net_model_1(X_data,input_dim,n_neur=10):
    W_1 = tf.Variable(tf.random_uniform([input_dim,n_neur]))
    b_1 = tf.Variable(tf.zeros([n_neur]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1) #activation function

    # layer 1 multiplying and adding bias then activation function
    W_2,b_2,layer_2 = hidden_layer(layer_1,n_neur)
   
    # layer 2 multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([n_neur,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)

    # O/p layer multiplying and adding bias then activation function

    # notice output layer has one node only since performing #regression

    return output
"""
neural_net_model is function applying 2 hidden layer feed forward neural net.

Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
These are variables with will be updated during training.

"""