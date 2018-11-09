import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from functions import read_data, neural_net_model_1, denormalize

#Parameters
name = "Cd_1"
alphaa = 'variable'
alpha_i = 0.002
n_neur = 10
optimizer = "gradient"
n_layers = 2 #change functions file

f = open("./data/"+name,"w+")
f.write(optimizer+"\n")
f.write(alphaa+"\n")
f.write("%5.8f \n" %alpha_i)
f.write("%5.8f \n" %n_neur)
f.write("%5.8f \n" %n_layers)

X_train, y_train_n, X_test, y_test_n, y_train, y_test = read_data() #Read de data
Cd_train = y_train_n[:,2]
Cd_test = y_test_n[:,2]
Cd_test_ok = y_test[:,2]
n_x = 16
xs = tf.placeholder("float")
ys = tf.placeholder("float")
alpha = tf.placeholder("float")
output = neural_net_model_1(xs,n_x, n_neur)
cost = tf.reduce_mean(tf.square(output-ys)) # our mean squared error cost function

if optimizer == "gradient":
    train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
elif optimizer == "Adam":
    train = tf.train.AdamOptimizer(alpha).minimize(cost)

c_t =[]
saver = tf.train.Saver()
start = time.time()

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    for i in range(25):
        for j in range(X_train.shape[0]):
            sess.run(train,feed_dict= {xs:X_train[j,:].reshape(1,n_x), ys:Cd_train[j], alpha: alpha_i})
            # Run train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:Cd_train}))
        print('Epoch :',i,'Cost :',c_t[i])
    end = time.time()
    delta_t = end-start
    norm_test_cost = sess.run(cost, feed_dict={xs:X_test,ys:Cd_test})
    print('Cost :',norm_test_cost)
    print('time [min]:', delta_t/60)
    pred = sess.run(output,feed_dict={xs:X_test})
    Mse0 = np.mean((pred-Cd_test)**2)
    print('Mse data w normalization:', Mse0)
    pred = denormalize(Cd_test_ok,pred)
    Mse = np.mean((pred-Cd_test_ok)**2)
    print('Mse data w/o normalization:', Mse)
    save_path = saver.save(sess,"./data/NN_"+name+".ckpt")
f.write("%5.8f \n"%Mse)
f.write("%5.8f \n"%Mse0)
f.write("%5.8f \n"%norm_test_cost)
f.write("%5.8f \n"%delta_t)
for item in c_t:
    f.write("%s " % item)
f.close()