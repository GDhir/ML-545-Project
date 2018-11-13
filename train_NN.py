#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from NN_airofil import NeuralAirfoil
from sklearn.preprocessing import MinMaxScaler

def read_data():
    """
    xy0, xy1, xy2 represent files corresponding to three outputs (CL, CM and CD)
    xy0:
    x: (x_geo (7 thickness, 7 cmaber) Ma, alpha),
    y: (CL), CM, CD,
    pCL/px, pCM/px, pCD/px
    """
    np.random.seed(0)
    xy0 = np.loadtxt('M0CFDdata.txt')
    xy1 = np.loadtxt('M1CFDdata.txt')
    xy2 = np.loadtxt('M2CFDdata.txt')
    xy = np.concatenate((np.concatenate((xy0, xy1)), xy2))
    np.random.shuffle(xy)
    dim = 16
    x = xy[:, :dim]
    y = xy[:, dim:dim+3]
    N_train = np.int(len(y[:,0])*0.8) #Number of train data
    y_train_ok = y[:N_train,:]
    y_test_ok = y[N_train:,:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x[:N_train,:])
    x_test = scaler.transform(x[N_train:,:])
    y_train_n = scaler.fit_transform(y[:N_train,:]) #normalized data
    y_test_n = scaler.transform(y[N_train:,:]) #normalized data

    return x_train, y_train_n, x_test, y_test_n, y_train_ok, y_test_ok

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

def train_NN(sess, model, X_train, y_train, num_epochs = 100, shuffle=False):
    sess.run(model.init_op)
    training_cost = []
    start = time.time()
    batch_size = int(X_train.shape[0]*0.05)
    for i in range(num_epochs):
        batch_generator = generate_batches(X_train, y_train, batch_size, shuffle)
        for batch_x, batch_y in batch_generator:
            feed = {model.X: batch_x, model.y: batch_y}
            _, cost = sess.run([model.optimizer, model.cost], feed_dict=feed)
        training_cost.append(cost)
        print('Epoch: ', i+1, 'Training cost: ', training_cost[-1])
    
    print('NN trained')
    end = time.time()
    total_time = end-start
    return training_cost, total_time

def generate_NN(sess, model, name, X_train, y_train, X_test, y_test, y_ok, y_test_ok,epochs=100, save=False, shuffle=False):
    train_cost, time = train_NN(sess, model, X_train, y_train, num_epochs=epochs, shuffle=shuffle)
    test_cost = sess.run(model.cost, feed_dict= {model.X:X_test, model.y: y_test})
    y_pred = sess.run(model.network, feed_dict={model.X: X_test})
    y_pred_ok = denormalize(y_ok, y_pred)
    print('Time [min]:', time/60)
    print('MSE', test_cost)

    fig1 = plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.semilogy(range(len(train_cost)),train_cost)
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.subplot(1,2,2)
    plt.plot(range(len(train_cost)),train_cost)
    plt.xlabel('Epoch')
    plt.show()

    fig2 = plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(range(len(y_test)), y_test, label= name +' Test')
    plt.plot(range(len(y_test)), y_pred, label=name + ' Predicted')
    plt.xlabel('Test Data')
    plt.ylabel(name)
    plt.title('Normalized data')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(range(len(y_test)), y_test_ok, label= name +' Test')
    plt.plot(range(len(y_test)), y_pred_ok, label=name + ' Predicted')
    plt.xlabel('Test Data')
    plt.title('Actual data')
    plt.legend()
    plt.show()
        
    fig3 = plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(range(len(y_test)), y_test-y_pred)
    plt.xlabel('Test Data')
    plt.ylabel('Error '+name)
    plt.title('Normalized data')
    plt.subplot(1,2,2)
    plt.plot(range(len(y_test)), y_test_ok-y_pred_ok)
    plt.xlabel('Test Data')
    plt.title('Actual data')
    plt.show()
    print('Relative Error %:',np.sqrt(np.sum((y_test_ok-y_pred_ok)**2))/np.sqrt(np.sum(y_test_ok**2))*100)
    # fig4 = plt.figure()
    # plt.plot(range(len(y_test)), (y_test_ok-y_pred_ok)/y_test_ok*100)
    # plt.xlabel('Test Data')
    # plt.title('Actual data')
    # plt.ylabel('% Error '+name)
    # plt.show()

   

    if save:
        save_path = model.saver.save(sess,"./data/NN_"+name+".ckpt")

        fig1.savefig('./data/training_cost'+name+'.pdf',bbox_inches='tight')
        fig2.savefig('./data/compare'+name+'.pdf',bbox_inches='tight')
        fig3.savefig('./data/error'+name+'.pdf',bbox_inches='tight')

        f = open("./data/"+name,"w+")
        f.write("%5.8f \n" %alpha)
        f.write("%5.8f \n" %n_neur)
        f.write("%5.8f \n" %n_layers)
        f.write("%5.8f \n"%test_cost)
        f.write("%5.8f \n"%time)
        for item in train_cost:
            f.write("%s " % item)
        f.close()

def denormalize(y, y_normalized):
    y = y.reshape(-1,1)
    y_normalized = y_normalized.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(y)
    new = scl.inverse_transform(y_normalized)
    return new


X_train, y_train, X_test, y_test, y_train_ok, y_test_ok = read_data() #Read de data

Cl_train = np.reshape(y_train[:,0],[-1,1])
Cl_test = np.reshape(y_test[:,0],[-1,1])
Cl_ok = np.reshape(y_train_ok[:,0],[-1,1])
Cl_test_ok = np.reshape(y_test_ok[:,0],[-1,1])

Cm_train = np.reshape(y_train[:,1],[-1,1])
Cm_test = np.reshape(y_test[:,1],[-1,1])
Cm_ok = np.reshape(y_train_ok[:,1],[-1,1])
Cm_test_ok = np.reshape(y_test_ok[:,1],[-1,1])

Cd_train = np.reshape(y_train[:,2],[-1,1])
Cd_test = np.reshape(y_test[:,2],[-1,1])
Cd_ok = np.reshape(y_train_ok[:,2],[-1,1])
Cd_test_ok = np.reshape(y_test_ok[:,2],[-1,1])

#%%
n_l = [1, 2, 3, 4, 5, 6, 7]
cost_t = []
time_t = []
n_neur = 80 
alpha = 0.001
K_folds = 3
batch_size_p = int(X_train.shape[0]/K_folds)+1
for i in range(len(n_l)):
    batch_generator = generate_batches(X_train, Cd_train, batch_size_p)
    cost_i = 0
    time_i = 0
    print('=================',n_l[i],'Layers =================')
    for batch_x, batch_y in batch_generator:
        model = NeuralAirfoil(N_hlayers=n_l[i], n_neur=n_neur, learning_rate=alpha)
        sess = tf.Session(graph = model.g)
        c , t = train_NN(sess, model, batch_x, batch_y, num_epochs=100)
        cost_i = cost_i + np.mean(np.asarray(c[-9:]))
        time_i = time_i + t
    cost_t.append(cost_i)
    time_t.append(time_i)

#%%
plt.figure()
plt.plot(n_l,np.asarray(cost_t)/K_folds)
plt.xlabel('Number of hidden layers')
plt.ylabel('Training Cost')
plt.savefig('./data/tc_nhl.pdf')

plt.figure()
plt.plot(n_l,np.asarray(time_t)/K_folds/60)
plt.xlabel('Number of hidden layers')
plt.ylabel('Time [min]')
plt.savefig('./data/time_nhl.pdf')

#%%

n_neur = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
# cost_t = []
# time_t = []
n_hl = 2
alpha = 0.001
K_folds = 3
cost_t = []
time_t = []
batch_size_p = int(X_train.shape[0]/K_folds)+1
for i in range(len(n_neur)):
    batch_generator = generate_batches(X_train, Cd_train, batch_size_p)
    cost_i = 0
    time_i = 0
    print('=================',n_neur[i],'Neurons =================')
    for batch_x, batch_y in batch_generator:
        model = NeuralAirfoil(N_hlayers=n_hl, n_neur=n_neur[i], learning_rate=alpha)
        sess = tf.Session(graph = model.g)
        c , t = train_NN(sess, model, batch_x, batch_y, num_epochs=100)
        cost_i = cost_i + np.mean(np.asarray(c[-10:]))
        time_i = time_i + t
    cost_t.append(cost_i)
    time_t.append(time_i)

#%%

plt.figure()
plt.plot(n_neur,np.asarray(cost_t)/K_folds)
plt.xlabel('Number of Neurons')
plt.ylabel('Training Cost')
plt.savefig('./data/tc_nneu.pdf')

plt.figure()
plt.plot(n_neur,np.asarray(time_t)/K_folds/60)
plt.xlabel('Number of Neurons')
plt.ylabel('Time [min]')
plt.savefig('./data/time_nneu.pdf')


#%%

alpha = .001
n_neur = 60
n_layers = 2 
epc = 500

model_Cd = NeuralAirfoil(N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha)
sess_Cd = tf.Session(graph = model_Cd.g)
generate_NN(sess_Cd, model_Cd, 'Cd', X_train, Cd_train, X_test, Cd_test, Cd_ok, Cd_test_ok, epc, True, True)

model_Cl = NeuralAirfoil(N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha)
sess_Cl = tf.Session(graph = model_Cl.g)
generate_NN(sess_Cl, model_Cl, 'Cl', X_train, Cl_train, X_test, Cl_test,Cl_ok, Cl_test_ok, epc, True, True)

model_Cm = NeuralAirfoil(N_hlayers=n_layers, n_neur=n_neur, learning_rate=alpha)
sess_Cm = tf.Session(graph = model_Cm.g)
generate_NN(sess_Cm, model_Cm, 'Cm', X_train, Cm_train, X_test, Cm_test,Cm_ok, Cm_test_ok, epc, True, True)
