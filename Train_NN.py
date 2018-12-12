#%%
import numpy as np
from NN_airfoil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler

def read_newdata():
    np.random.seed(0)
    xy0 = np.loadtxt('new_data1.txt')
    xy1 = np.loadtxt('new_data2.txt')
    xy2 = np.loadtxt('new_data3.txt')
    xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))
    np.random.shuffle(xy)
    dim = 281+2 #281 points + Mach number + angle of attack 
    x = xy[:, :dim]
    y = xy[:, dim:dim+3]
    N_train = np.int(len(y[:,0])*0.8) #Number of train data
    y_train_ok = y[:N_train,:]
    y_test_ok = y[N_train:,:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x[:N_train,:]) #normalize the X_train
    x_test = scaler.transform(x[N_train:,:]) #normalize the X_test
    y_train_n = scaler.fit_transform(y[:N_train,:]) #normalized data
    y_test_n = scaler.transform(y[N_train:,:]) #normalized data
    return x_train, y_train_n, x_test, y_test_n, y_train_ok, y_test_ok


X_train, y_train, X_test, y_test, y_train_ok, y_test_ok = read_newdata() #Read de data

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
x_dim  = 281 #y points
epc = 10
save = False
show = False

#create NN for Cd
model_Cd = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha, num_epochs=epc)

#Train the NN for Cd
model_Cd.train_NN(X_train, Cd_train, X_test, Cd_test, Cd_ok, Cd_test_ok)
#generate the plots for Cd
model_Cd.generate_plot('Cd', show=show, save=save)
print('Relative error test %',model_Cd.R_error_test[-1])
print('Relative error train %',model_Cd.R_error_train[-1])

#%%
# model_Cm = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha, num_epochs=epc)
# model_Cm.train_NN(X_train, Cm_train, X_test, Cm_test, Cm_ok, Cm_test_ok)
# model_Cm.generate_plot('Cm', show=show, save=save)
# print('Relative error test %',model_Cm.R_error_test[-1])
# print('Relative error train %',model_Cm.R_error_train[-1])


# model_Cl = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha, num_epochs=epc)
# model_Cl.train_NN(X_train, Cl_train, X_test, Cl_test, Cl_ok, Cl_test_ok)
# model_Cl.generate_plot('Cl', show=show, save=save)
# print('Relative error test %',model_Cl.R_error_test[-1])
# print('Relative error train %',model_Cl.R_error_train[-1])