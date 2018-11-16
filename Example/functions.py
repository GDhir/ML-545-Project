import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_data():
    df = pd.read_csv('data.csv') # read data set using pandas
    # print(df.info()) # Overview of dataset
    df = df.drop(['Date'],axis=1) # Drop Date feature
    df = df.dropna(inplace=False)  # Remove all nan entries.
    df = df.drop(['Adj Close','Volume'],axis=1) # Drop Adj close and volume feature
    df_train = df[:1000]    # 60% training data and 40% testing data
    df_test = df[1000:]
    scaler = MinMaxScaler() # For normalizing dataset
    # We want to predict Close value of stock 
    X_train = scaler.fit_transform(df_train.drop(['Close'],axis=1).values)
    y_train = scaler.fit_transform(df_train['Close'].values.reshape(-1,1))
    # # y is output and x is features.
    X_test = scaler.fit_transform(df_test.drop(['Close'],axis=1).values)
    y_test = scaler.fit_transform(df_test['Close'].values.reshape(-1,1))
    return X_train, y_train, X_test, y_test, df_test, df_train

def denormalize(df,norm_data):
    df = df['Close'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new

def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.elu(layer_1) #activation function

    # layer 1 multiplying and adding bias then activation function

    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.elu(layer_2) #activation function

    # layer 2 multiplying and adding bias then activation function

    W_O = tf.Variable(tf.random_uniform([10,1]))
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