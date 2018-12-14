#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# with open('cd_train_history.csv') as inp:
# 	i = 0
# 	X = [0,0,0,0]
# 	for line in inp:
# 		a = inp.readline()
# 		b = a.strip("\n").split(" ")
# 		X = np.vstack([X, b])

# X = np.delete(X, (0), axis=0)

# print(X)
# Z = np.zeros((4,24,45))
# # np.reshape(X, (4, 25, 45))

# print(len(X)/26)

# # print(X[0,0,:])

# X = np.loadtxt(open('cd_train_history.csv', "rb"))

# The data to load
f = "cd_train_history_multilayer8.csv"

# # Take every N-th (in this case 10th) row
# n = 25

# # Count the lines or use an upper bound
# num_lines = sum(1 for l in open(f))

# # The row indices to skip - make sure 0 is not included to keep the header!
# skip_idx = [x for x in range(1, num_lines) if x % n != 0]

# Read the data

def count_neurons(row):
	tsum = 0
	for i in range(0,int(row[2])):
		tsum = tsum + row[3+i]
	return tsum

data = pd.read_csv(f)
data['NNeurons'] = data.apply(lambda row: count_neurons (row), axis=1)
data.plot(kind='scatter', x='Generation', y='MSE', logy=True)
data.plot(kind='scatter', x='Generation', y='Time', logy=True)
data.plot(kind='scatter', x='Generation', y='f', logy=True)
data.plot(kind='scatter', x='Generation', y='NNeurons', logy=True)
plt.show()
print(data)