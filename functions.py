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
    return x_train, y_train, x_test, y_test 


