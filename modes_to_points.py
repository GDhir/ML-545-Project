#%%
import numpy as np
import matplotlib.pyplot as plt

xy0 = np.loadtxt('M0CFDdata.txt')
xy1 = np.loadtxt('M1CFDdata.txt')
xy2 = np.loadtxt('M2CFDdata.txt')
xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))

basedata = np.loadtxt('basis.txt')
x = basedata[0,:].copy()
U = basedata[1:,:].copy() 
N_data = xy.shape[0]
N_points = U.shape[1] 
new_data = np.zeros((N_data, N_points + 5)) 

for i in range(N_data):
    new_data[i,0:N_points] = np.dot(xy[i,0:14],U)
new_data[:,N_points:N_points+5] = xy[:,14:19]
np.savetxt('new_data.txt',new_data)
np.savetxt('x_geom.txt',x)
