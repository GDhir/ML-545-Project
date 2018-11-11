#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from NN_airofil import *
from sklearn.preprocessing import MinMaxScaler

def read_data():
    """
    xy0, xy1, xy2 represent files corresponding to three outputs (CL, CM and CD)
    xy0:
    x: (x_geo (7 thickness, 7 cmaber) Ma, alpha),
    y: (CL), CM, CD,
    pCL/px, pCM/px, pCD/px
    """
    xy0 = np.loadtxt('M0CFDdata.txt')
    xy1 = np.loadtxt('M1CFDdata.txt')
    xy2 = np.loadtxt('M2CFDdata.txt')
    xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))
    dim = 16
    x = xy[:, :dim]
    y = xy[:, dim:dim+3]
    scaler = MinMaxScaler()
    N_train = np.int(len(y[:,0])*0.8) #Number of train data
    x_train = scaler.fit_transform(x[:N_train,:])
    x_test = scaler.transform(x[N_train:,:])
    y_train = y[:N_train,:]
    y_test = y[N_train:,:]
    return x_train, y_train, x_test, y_test 


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

def train_NN(sess, model, X_train, y_train, num_epochs = 100):
    sess.run(model.init_op)
    training_cost = []
    start = time.time()
    batch_size = int(X_train.shape[0]*0.05)
    for i in range(num_epochs):
        batch_generator = generate_batches(X_train, y_train, batch_size)
        for batch_x, batch_y in batch_generator:
            feed = {model.X: batch_x, model.y: batch_y}
            _, cost = sess.run([model.optimizer, model.cost], feed_dict=feed)
            training_cost.append(cost)
        print('Epoch: ', i+1, 'Training cost: ', training_cost[-1])
    
    print('NN trained')
    end = time.time()
    total_time = end-start
    return training_cost, total_time

def predict_output(sess, model, X_test):
    y_pred = sess.run(model.network, feed_dict={model.X: X_test})
    return y_pred

#Parameters
name = "Cd_1"
alpha_i = .001
n_neur = 10
n_layers = 2 #change functions file


X_train, y_train, X_test, y_test = read_data() #Read de data
Cd_train = np.reshape(y_train[:,2],[-1,1])
Cd_test = np.reshape(y_test[:,2],[-1,1])

#%%
model = NeuralAirfoil(N_hlayers=n_layers, n_neur=n_neur)
sess = tf.Session(graph = model.g)
train_cost, time = train_NN(sess, model, X_train, Cd_train)
test_cost = sess.run(model.cost, feed_dict= {model.X:X_test, model.y: Cd_test})
Cd_pred = predict_output(sess, model, X_test)
save_path = model.saver.save(sess,"./data/NN_"+name+".ckpt")

#%%
print('Time [min]:', time/60)
print('MSE', test_cost)
plt.figure()
plt.plot(range(len(train_cost)),train_cost)
plt.xlabel('Iteration')
plt.ylabel('Training Cost')
plt.show()

plt.figure()
plt.plot(range(len(Cd_test)), Cd_test, label=r'$C_d$ Test')
plt.plot(range(len(Cd_test)), Cd_pred, label=r'$C_d$Predicted')
plt.legend()
plt.show()


f = open("./data/"+name,"w+")
f.write("%5.8f \n" %alpha_i)
f.write("%5.8f \n" %n_neur)
f.write("%5.8f \n" %n_layers)
f.write("%5.8f \n"%test_cost)
f.write("%5.8f \n"%time)
for item in train_cost:
    f.write("%s " % item)
f.close()