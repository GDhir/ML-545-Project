#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NN_airofil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler


l_rate = .001
n_neur = 60
n_layers = 2
x_dim  = 16#281+2
y_dim = 1
epc = 300
save = False

model = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur, learning_rate=l_rate, num_epochs=epc)

saver_Cd = model.saver
session_Cd = tf.Session(graph=model.g)
saver_Cd.restore(session_Cd,'./data/NN_Cd.ckpt')

saver_Cl = model.saver
session_Cl = tf.Session(graph=model.g)
saver_Cl.restore(session_Cl,'./data/NN_Cl.ckpt')

saver_Cm = model.saver
session_Cm = tf.Session(graph=model.g)
saver_Cm.restore(session_Cm,'./data/NN_Cm.ckpt')


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

pred_Cd = session_Cd.run(model.network,feed_dict={model.X:x_eval})
pred_Cd = denormalize_y(pred_Cd, 1)

pred_Cl = session_Cl.run(model.network,feed_dict={model.X:x_eval})
pred_Cl = denormalize_y(pred_Cl, 0)

pred_Cm = session_Cm.run(model.network,feed_dict={model.X:x_eval})
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

        

