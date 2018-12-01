#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NN_airofil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as scopt

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
    x_t = np.zeros(x_dim)
    x_t[0:14] = x[0:14]
    x_t = denormalize_x(x_t)
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
Mach = 0.45
Cl_req = np.asscalar(normalize_y(np.array([0.5]),0))

model = NeuralAirfoil(x_dim = x_dim, N_hlayers=n_layers, n_neur=n_neur)

saver_Cd = model.saver
session_Cd = tf.Session(graph=model.g)
saver_Cd.restore(session_Cd, './data/NN_Cd.ckpt')

saver_Cl = model.saver
session_Cl = tf.Session(graph=model.g)
saver_Cl.restore(session_Cl, './data/NN_Cl.ckpt')

def Cd(x_in):
    x = np.zeros(x_dim)
    x[0:14] = x_in[0:14]
    x[14] = Mach_n
    x[15] = x_in[14]
    x = x.reshape(1,x_dim)
    pred = session_Cd.run(model.network,feed_dict={model.X:x})
    return np.asscalar(pred)

def Cl(x_in):
    x = np.zeros(x_dim)
    x[0:14] = x_in[0:14]
    x[14] = Mach_n
    x[15] = x_in[14]
    x = x.reshape(1,x_dim) 
    pred = session_Cl.run(model.network,feed_dict={model.X:x})
    return np.asscalar(pred)

def c1(x):
    return thickness(x)[0]-thickness_constraint[0]
def c2(x):
    return thickness(x)[1]-thickness_constraint[1]
def c3(x):
    return thickness(x)[2]-thickness_constraint[2]
def c4(x):
    return thickness(x)[3]-thickness_constraint[3]
def c5(x):
    return thickness(x)[4]-thickness_constraint[4]
def c6(x):
    return Cl(x) - Cl_req

con1 = {'type': 'ineq', 'fun': c1}
con2 = {'type': 'ineq', 'fun': c2}
con3 = {'type': 'ineq', 'fun': c3}
con4 = {'type': 'ineq', 'fun': c4}
con5 = {'type': 'ineq', 'fun': c5}
con6 = {'type': 'eq', 'fun': c6}

cons = (con1,con2,con3,con4,con5,con6)
eqcons = [c6]
ineqcons  = [c1,c2,c3,c4,c5]
#%%
x0_t = np.zeros(16)
x0_t[:14] = airfoil
x0_t[14] = Mach
x0_t[15] = 2
x0_t = normalize_x(x0_t)
x0 = np.zeros(15)
x0[0:14] = x0_t[0,:14]
x0[14] = x0_t[0,15]
Mach_n = x0_t[0,14]
print(Cd(x0))
print(denormalize_y(np.array([Cl(x0)]),0))
tol = 1e-8

#%%
sol = scopt.minimize(Cd, x0, method='SLSQP',constraints =cons ,tol=tol, options={'disp': True})
# sol = scopt.fmin_slsqp(Cd, x0, eqcons=eqcons, ieqcons = ineqcons)

print(sol.success)
print(denormalize_y(np.array([sol.fun]),1))
print(denormalize_y(np.array([Cd(x0)]),1))
sol_opt = np.zeros(16)
sol_opt[0:14] = sol.x[0:14]
sol_opt[15] = sol.x[14]
sol_opt[14] = Mach_n
sol_opt = denormalize_x(sol_opt)
print(sol_opt)
print(airfoil)
#%%
y_opt = np.dot(sol_opt[0,0:14],U)
plt.figure()
plt.axis('equal')
plt.plot(x_geom,y_opt)
plt.plot(x_geom,y_airfoil,'--')


