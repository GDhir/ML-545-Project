#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NN_airofil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler

def read_data():
    """
    xy0, xy1, xy2 represent files corresponding to three outputs (CL, CM and CD)
    xy0:
    x: (x_geo (7 thickness, 7 cmaber) Ma, alpha),
    y: (CL), CM, CD,
    pCL/px, pCM/px, pCD/px (?? not sure about the order)
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
    x_train = scaler.fit_transform(x[:N_train,:]) #normalize the X_train
    x_test = scaler.transform(x[N_train:,:]) #normalize the X_test
    y_train_n = scaler.fit_transform(y[:N_train,:]) #normalized data
    y_test_n = scaler.transform(y[N_train:,:]) #normalized data
    return x_train, y_train_n, x_test, y_test_n, y_train_ok, y_test_ok

X_train, y_train, X_test, y_test, y_train_ok, y_test_ok = read_data() #Read de data

Cd_train = np.reshape(y_train[:,2],[-1,1])
Cd_test = np.reshape(y_test[:,2],[-1,1])
Cd_test_ok = np.reshape(y_test_ok[:,2],[-1,1])
Cd_train_ok = np.reshape(y_train_ok[:,2],[-1,1])


l_rate = .001
n_neur = 60
n_layers = 2
x_dim  = 16#281+2
y_dim = 1
epc = 300
save = False

def Cd(x):
        model_Cd = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=l_rate, num_epochs=epc)
        saver = model_Cd.saver
        with tf.Session(graph=model_Cd.g) as session:   
                #Load NN
                saver.restore(session, './data/NN_Cd.ckpt')
                # Evalute test data
                pred = session.run(model_Cd.network,feed_dict={model_Cd.X:x})
        session.close()
        return pred

def Cl(x):
        model_Cl = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=l_rate, num_epochs=epc)
        saver = model_Cl.saver
        with tf.Session(graph=model_Cl.g) as session:   
                #Load NN
                saver.restore(session, './data/NN_Cl.ckpt')
                # Evalute test data
                pred = session.run(model_Cl.network,feed_dict={model_Cl.X:x})
        session.close()
        return pred

def Cm(x):
        model_Cm = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=l_rate, num_epochs=epc)
        saver = model_Cm.saver
        with tf.Session(graph=model_Cm.g) as session:   
                #Load NN
                saver.restore(session, './data/NN_Cm.ckpt')
                # Evalute test data
                pred = session.run(model_Cm.network,feed_dict={model_Cm.X:x})
        session.close()
        return pred


def denormalize_y(y, var):
        """
        var: 0 for Cl, 1 for Cd, 2 for Cm
        """
        y_tr = np.loadtxt('y_train.txt')
        scaler = MinMaxScaler()
        scaler.fit(y_tr[:,var].reshape(-1,1))
        y_den = scaler.inverse_transform(y.reshape(-1,1))
        return y_den

def normalize_x(x):
        x_tr = np.loadtxt('x_train.txt')
        scaler = MinMaxScaler()
        scaler.fit(x_tr)
        x_transf = scaler.transform(x)
        return x_transf


#%%


airfoil = np.loadtxt('clarky.coef')
Ma = 0.45
alpha = np.linspace(-2, 6, num=20)
x_eval = np.zeros((len(alpha),16))
x_eval[:,0:14] = airfoil
x_eval[:,14] = Ma
pred_Cd = []
for i in range(len(alpha)):
        x_eval[i,15] = alpha[i]
x_eval = normalize_x(x_eval)

pred_Cd = Cd(x_eval)
pred_Cd = denormalize_y(pred_Cd, 1)

pred_Cl = Cl(x_eval)
pred_Cl = denormalize_y(pred_Cl, 0)

pred_Cm = Cm(x_eval)
pred_Cm = -denormalize_y(pred_Cm, 2)

f = plt.figure(figsize=(25,5))
f1 = f.add_subplot(132)
f2 = f.add_subplot(131)
f3 = f.add_subplot(133)

f1.plot(alpha,pred_Cd*1E4,label="NN")
f1.set_ylabel(r'$C_d$ (Counts)',fontsize=20)
f1.set_xlabel(r'$\alpha$',fontsize=20)
f1.set_xlim([min(alpha),max(alpha)])
f1.grid()


f2.plot(alpha,pred_Cl,label="NN")
f2.set_ylabel(r'$C_l$',fontsize=20)
f2.set_xlabel(r'$\alpha$',fontsize=20)
f2.set_xlim([min(alpha),max(alpha)])
f2.grid()

f3.plot(alpha,pred_Cm,label="NN")
f3.set_ylabel(r'$C_m$',fontsize=20)
f3.set_xlabel(r'$\alpha$',fontsize=20)
f3.set_ylim([-0.25,0])
f3.set_xlim([min(alpha),max(alpha)])
f3.grid()

f.savefig('./data/ClarkY.pdf',bbox_inches='tight')

        

