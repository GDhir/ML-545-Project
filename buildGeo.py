import numpy as np
samples = np.loadtxt('sample.txt')
basedata = np.loadtxt('basis.txt')
x = basedata[0,:].copy()
U = basedata[1:,:].copy()
nbasis = 14
npts = basedata.shape[1]

basey = np.dot(samples,U)
f = open('new.plt','w')
for i in xrange(npts):
    f.write('%.15f %.15f\n'%(x[i],basey[i]))
f.close()
