#%%
import numpy as np
from NN_airofil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler

def read_data():
    """
    xy0, xy1, xy2 represent files corresponding to three outputs (CL, CM and CD)
    xy0:
    x: (x_geo (7 thickness, 7 cmaber) Ma, alpha),
    y: (CL), CD, CM,
    pCL/px, pCM/px, pCD/px (??? not sure about the order)
    """
    np.random.seed(0)
    xy0 = np.loadtxt('M0CFDdata.txt')
    xy1 = np.loadtxt('M1CFDdata.txt')
    xy2 = np.loadtxt('M2CFDdata.txt')
    xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))
    #shuffle the data, because it has some pattern
    np.random.shuffle(xy)
    dim = 16
    x = xy[:, :dim]
    y = xy[:, dim:dim+3]
    N_train = np.int(len(y[:,0])*0.8) #Number of train data
    y_train_ok = y[:N_train,:]
    y_test_ok = y[N_train:,:]
    scaler = MinMaxScaler()
    x_train = x[:N_train,:]
    np.savetxt('x_train.txt',x_train)
    np.savetxt('y_train.txt', y_train_ok)
    x_train = scaler.fit_transform(x_train) #normalize the X_train
    x_test = scaler.transform(x[N_train:,:]) #normalize the X_test
    y_train_n = scaler.fit_transform(y[:N_train,:]) #normalized data
    y_test_n = scaler.transform(y[N_train:,:]) #normalized data
    return x_train, y_train_n, x_test, y_test_n, y_train_ok, y_test_ok


X_train, y_train, X_test, y_test, y_train_ok, y_test_ok = read_data() #Read de data

Cl_train = np.reshape(y_train[:,0],[-1,1])
Cl_test = np.reshape(y_test[:,0],[-1,1])
Cl_ok = np.reshape(y_train_ok[:,0],[-1,1])
Cl_test_ok = np.reshape(y_test_ok[:,0],[-1,1])

Cm_train = np.reshape(y_train[:,2],[-1,1])
Cm_test = np.reshape(y_test[:,2],[-1,1])
Cm_ok = np.reshape(y_train_ok[:,2],[-1,1])
Cm_test_ok = np.reshape(y_test_ok[:,2],[-1,1])

Cd_train = np.reshape(y_train[:,1],[-1,1])
Cd_test = np.reshape(y_test[:,1],[-1,1])
Cd_ok = np.reshape(y_train_ok[:,1],[-1,1])
Cd_test_ok = np.reshape(y_test_ok[:,1],[-1,1])

#%%

alpha = .001
n_neur = 60
n_layers = 2
x_dim  = 16#281+2
epc = 1000
save = False

#create NN for Cd
model_Cd = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha, num_epochs=epc)
#Train the NN for Cd
model_Cd.train_NN(X_train, Cd_train, X_test, Cd_test, Cd_ok, Cd_test_ok)
#generate the plots for Cd
model_Cd.generate_plot('Cd', show=True, save=save)
print('Relative error test %',model_Cd.R_error_test[-1])
print('Relative error train %',model_Cd.R_error_train[-1])

#%%
model_Cm = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha, num_epochs=epc)
model_Cm.train_NN(X_train, Cm_train, X_test, Cm_test, Cm_ok, Cm_test_ok)
model_Cm.generate_plot('Cm', show=True, save=save)
print('Relative error test %',model_Cm.R_error_test[-1])
print('Relative error train %',model_Cm.R_error_train[-1])

alpha = .01

model_Cl = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha, num_epochs=epc)
model_Cl.train_NN(X_train, Cl_train, X_test, Cl_test, Cl_ok, Cl_test_ok)
model_Cl.generate_plot('Cl', show=True, save=save)
print('Relative error test %',model_Cl.R_error_test[-1])
print('Relative error train %',model_Cl.R_error_train[-1])