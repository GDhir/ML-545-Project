#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from NN_airofil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler

def read_data():
    """
    xy0, xy1, xy2 represent files corresponding to three outputs (CL, CM and CD)
    xy0:
    x: (x_geo (7 thickness, 7 cmaber) Ma, alpha),
    y: (CL), CM, CD,
    pCL/px, pCM/px, pCD/px
    """
    np.random.seed(0)
    xy0 = np.loadtxt('M0CFDdata.txt')
    xy1 = np.loadtxt('M1CFDdata.txt')
    xy2 = np.loadtxt('M2CFDdata.txt')
    xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))
    np.random.shuffle(xy)
    dim = 16
    x = xy[:, :dim]
    y = xy[:, dim:dim+3]
    N_train = np.int(len(y[:,0])*0.8) #Number of train data
    y_train_ok = y[:N_train,:]
    y_test_ok = y[N_train:,:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x[:N_train,:])
    x_test = scaler.transform(x[N_train:,:])
    y_train_n = scaler.fit_transform(y[:N_train,:]) #normalized data
    y_test_n = scaler.transform(y[N_train:,:]) #normalized data

    return x_train, y_train_n, x_test, y_test_n, y_train_ok, y_test_ok

def generate_batches(X, y, batch_size, shuffle=False):
    X_copy = np.array(X) 
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:,:-1]
        y_copy = data[:,-1].reshape((-1,1))
    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size,:], y_copy[i:i+batch_size])

X_train, y_train, X_test, y_test, y_train_ok, y_test_ok = read_data() #Read de data

Cl_train = np.reshape(y_train[:,0],[-1,1])
Cl_test = np.reshape(y_test[:,0],[-1,1])
Cl_ok = np.reshape(y_train_ok[:,0],[-1,1])
Cl_test_ok = np.reshape(y_test_ok[:,0],[-1,1])

Cm_train = np.reshape(y_train[:,1],[-1,1])
Cm_test = np.reshape(y_test[:,1],[-1,1])
Cm_ok = np.reshape(y_train_ok[:,1],[-1,1])
Cm_test_ok = np.reshape(y_test_ok[:,1],[-1,1])

Cd_train = np.reshape(y_train[:,2],[-1,1])
Cd_test = np.reshape(y_test[:,2],[-1,1])
Cd_ok = np.reshape(y_train_ok[:,2],[-1,1])
Cd_test_ok = np.reshape(y_test_ok[:,2],[-1,1])

#%%
n_l = [1, 2, 3, 4, 5, 6, 7]
cost_t = []
time_t = []
n_neur = 80 
alpha = 0.001
K_folds = 3
batch_size_p = int(X_train.shape[0]/K_folds)+1
for i in range(len(n_l)):
    batch_generator = generate_batches(X_train, Cd_train, batch_size_p)
    cost_i = 0
    time_i = 0
    print('=================',n_l[i],'Layers =================')
    for batch_x, batch_y in batch_generator:
        model = NeuralAirfoil(N_hlayers=n_l[i], n_neur=n_neur, learning_rate=alpha)
        sess = tf.Session(graph = model.g)
        c , t = train_NN(batch_x, batch_y, num_epochs=100)
        cost_i = cost_i + np.mean(np.asarray(c[-9:]))
        time_i = time_i + t
    cost_t.append(cost_i)
    time_t.append(time_i)

#%%
plt.figure()
plt.plot(n_l,np.asarray(cost_t)/K_folds)
plt.xlabel('Number of hidden layers')
plt.ylabel('Training Cost')
plt.savefig('./data/tc_nhl.pdf')

plt.figure()
plt.plot(n_l,np.asarray(time_t)/K_folds/60)
plt.xlabel('Number of hidden layers')
plt.ylabel('Time [min]')
plt.savefig('./data/time_nhl.pdf')

#%%

n_neur = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
# cost_t = []
# time_t = []
n_hl = 2
alpha = 0.001
K_folds = 3
cost_t = []
time_t = []
batch_size_p = int(X_train.shape[0]/K_folds)+1
for i in range(len(n_neur)):
    batch_generator = generate_batches(X_train, Cd_train, batch_size_p)
    cost_i = 0
    time_i = 0
    print('=================',n_neur[i],'Neurons =================')
    for batch_x, batch_y in batch_generator:
        model = NeuralAirfoil(N_hlayers=n_hl, n_neur=n_neur[i], learning_rate=alpha)
        model.train_NN(sess, model, batch_x, batch_y, num_epochs=100)
        cost_i = cost_i + np.mean(np.asarray(c[-10:]))
        time_i = time_i + t
    cost_t.append(cost_i)
    time_t.append(time_i)

#%%

plt.figure()
plt.plot(n_neur,np.asarray(cost_t)/K_folds)
plt.xlabel('Number of Neurons')
plt.ylabel('Training Cost')
plt.savefig('./data/tc_nneu.pdf')

plt.figure()
plt.plot(n_neur,np.asarray(time_t)/K_folds/60)
plt.xlabel('Number of Neurons')
plt.ylabel('Time [min]')
plt.savefig('./data/time_nneu.pdf')


