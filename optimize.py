#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NN_airfoil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler
import pyoptsparse
from pyoptsparse import Optimization

basedata = np.loadtxt('basis.txt') 
x_geom = basedata[0,:].copy()
U = basedata[1:,:].copy()

airfoil = np.loadtxt('naca4412.coef')
y_airfoil = np.dot(airfoil,U)

x_tr = np.loadtxt('x_train.txt')
y_tr = np.loadtxt('y_train.txt')

x_constraint = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
thickness_constraint = [] 
index_constraint = []
for i in range(len(x_constraint)):
    x_var = np.abs(x_geom-x_constraint[i])
    index = np.where(x_var==np.amin(x_var))
    index = np.array([min(index[0]), max(index[0])])
    index_constraint.append(index)
    thickness_constraint.append(0.9*(y_airfoil[index[0]]-y_airfoil[index[1]]))

def thickness(x):
    x_t = denormalize_x(x)
    x_f  = x_t[0,0:14]
    x_new = np.dot(x_f,U)
    thickness = []
    for i in range(len(x_constraint)):
        thickness.append(x_new[index_constraint[i][0]]-x_new[index_constraint[i][1]])
    return np.asscalar(thickness[0]),np.asscalar(thickness[1]),np.asscalar(thickness[2]),np.asscalar(thickness[3]),np.asscalar(thickness[4])

def normalize_x(x):
    x = x.reshape(1,x_dim)
    scaler = MinMaxScaler()
    scaler.fit(x_tr)
    x_transf = scaler.transform(x)
    return x_transf

def denormalize_x(x):
    x = x.reshape(1,x_dim)
    scaler = MinMaxScaler()
    scaler.fit(x_tr)
    x_transf = scaler.inverse_transform(x)
    return x_transf

def denormalize_y(y, var):
    """
    var: 0 for Cl, 1 for Cd, 2 for Cm
    """
    scaler = MinMaxScaler()
    scaler.fit(y_tr[:,var].reshape(-1,1))
    y_den = scaler.inverse_transform(y.reshape(-1,1))
    return y_den

def normalize_y(y, var):
    """
    var: 0 for Cl, 1 for Cd, 2 for Cm
    """
    scaler = MinMaxScaler()
    scaler.fit(y_tr[:,var].reshape(-1,1))
    y_den = scaler.transform(y.reshape(-1,1))
    return y_den

n_neur = 60
n_layers = 2
x_dim  = 16#281+2


model = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur)

saver_Cd = model.saver
session_Cd = tf.Session(graph=model.g)
saver_Cd.restore(session_Cd, './data/NN_Cd.ckpt')

saver_Cl = model.saver
session_Cl = tf.Session(graph=model.g)
saver_Cl.restore(session_Cl, './data/NN_Cl.ckpt')

def Cd(x):
    pred = session_Cd.run(model.network,feed_dict={model.X:x})
    return np.asscalar(pred)

def Cl(x):
    pred = session_Cl.run(model.network,feed_dict={model.X:x})
    return np.asscalar(pred)

def grad_thick_con():
    der_c = np.zeros((5,14))
    maxi = np.amax(x_tr, axis=0)
    mini = np.amin(x_tr, axis=0)
    for i in range(5):
        index = index_constraint[i]
        for j in range(14):
            der_c[i,j] = (U[j,index[0]]- U[j,index[1]])*(maxi[j]-mini[j])
    return der_c
#%%
def objfunc(xdict):
    modes = xdict['modes']
    alpha = xdict['alpha']
    x = np.zeros(x_dim)
    x[0:14] = modes
    x[14] = Mach_n
    x[15] = alpha
    x = x.reshape(1,x_dim) 
    funcs = {}
    funcs['obj'] = Cd(x)
    funcs['thick_0.1'] = thickness(x)[0]
    funcs['thick_0.3'] = thickness(x)[1]
    funcs['thick_0.5'] = thickness(x)[2]
    funcs['thick_0.7'] = thickness(x)[3]
    funcs['thick_0.9'] = thickness(x)[4]
    funcs['Cl'] = Cl(x)
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    modes = xdict['modes']
    alpha = xdict['alpha']
    x = np.zeros(x_dim)
    x[0:14] = modes
    x[14] = Mach_n
    x[15] = alpha
    x = x.reshape(1,x_dim) 
    funcsSens = {}
    grad_Cd = session_Cd.run(model.gradient_NN,feed_dict={model.X:x})[0]
    grad_Cl = session_Cl.run(model.gradient_NN,feed_dict={model.X:x})[0]
    funcsSens['obj','modes'] = grad_Cd[0,0:14]
    funcsSens['obj', 'alpha'] = grad_Cd[0,15]
    funcsSens['Cl','modes'] = grad_Cl[0,0:14]
    funcsSens['Cl', 'alpha'] = grad_Cl[0,15]
    funcsSens['thick_0.1','modes'] = grad_thick_con()[0,:]
    funcsSens['thick_0.1','alpha'] = [0]
    funcsSens['thick_0.3','modes'] = grad_thick_con()[1,:]
    funcsSens['thick_0.3','alpha'] = [0]
    funcsSens['thick_0.5','modes'] = grad_thick_con()[2,:]
    funcsSens['thick_0.5','alpha'] = [0]
    funcsSens['thick_0.7','modes'] = grad_thick_con()[3,:]
    funcsSens['thick_0.7','alpha'] = [0]
    funcsSens['thick_0.9','modes'] = grad_thick_con()[4,:]
    funcsSens['thick_0.9','alpha'] = [0]
    fail = False
    return funcsSens, fail

Mach = 0.45
Cl_i = np.array([0.5])
Cl_req = np.asscalar(normalize_y(Cl_i,0))
x0_t = np.zeros(16)
x0_t[14] = Mach
x0_t = normalize_x(x0_t)
Mach_n = x0_t[0,14]
low_alpha = None
up_alpha = None

optProb = Optimization('naca4412', objfunc)

optProb.addVarGroup('modes', 14, 'c', lower=None, upper=None, value=.5)
optProb.addVar('alpha', 'c', lower=low_alpha, upper=up_alpha, value=.5)

optProb.addCon('thick_0.1', lower=thickness_constraint[0], upper=None)
optProb.addCon('thick_0.3', lower=thickness_constraint[1], upper=None)
optProb.addCon('thick_0.5', lower=thickness_constraint[2], upper=None)
optProb.addCon('thick_0.7', lower=thickness_constraint[3], upper=None)
optProb.addCon('thick_0.9', lower=thickness_constraint[4], upper=None)
optProb.addCon('Cl', lower=Cl_req, upper=Cl_req)

optProb.addObj('obj')

print(optProb)
#%%
opt = pyoptsparse.SLSQP()
sol = opt(optProb, sens=sens)
print(sol)
#%%
modes_opt = sol.xStar['modes']
alpha_opt = sol.xStar['alpha']
Cd_opt = denormalize_y(sol.fStar,1)
x0 = np.zeros((1,x_dim))
x0[0,0:14] = airfoil
x0[0,14] = Mach 
x0_n = normalize_x(x0)
Cd_0 = Cd(x0_n)
print(Cd_opt, Cd_0)
x_opt = np.zeros((1,x_dim))
x_opt[0,0:14] = modes_opt
x_opt[0,15] = alpha_opt
x_opt = denormalize_x(x_opt)
print('alpha', x_opt[0,15])
y_opt = np.dot(x_opt[0,0:14],U)
plt.figure()
plt.axis('equal')
plt.plot(x_geom,y_airfoil,'--',label = 'Base airfoil')
plt.plot(x_geom,y_opt,label='Optimized')
plt.legend()


