import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import read_data, neural_net_model_1, denormalize

X_train, y_train_n, X_test, y_test_n,  y_train, y_test= read_data()
Cd_train = np.reshape(y_train_n[:,2],[-1,1])
Cd_test = np.reshape(y_test_n[:,2],[-1,1])
Cd_test_ok = np.reshape(y_test[:,2],[-1,1])
Cd_train_ok = np.reshape(y_train[:,2],[-1,1])
xs = x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]],name="xs") 
ys =  tf.placeholder(tf.float32, shape=[None, 1],name="ys")
output = neural_net_model_1(xs,16)
cost = tf.reduce_mean(tf.square(output-ys))
saver = tf.train.Saver()
with tf.Session() as session:
    #Load NN
    saver.restore(session, './data/NN_Cd_1.ckpt')
    # Evalute test data
    pred_n = session.run(output,feed_dict={xs:X_test})
    Mse1 = np.mean((pred_n-y_test_n[:,2])**2)
    Mse2 = session.run(cost, feed_dict={xs:X_test,ys:Cd_test})
    print(Mse1,Mse2)
    pred = denormalize(Cd_train_ok,pred_n)
    
    Mse = np.mean((pred-Cd_test_ok)**2)
    print(Mse)
    #Plot results
    plt.figure()
    plt.plot(range(Cd_test_ok.shape[0]),pred,label="Predicted Data")
    plt.plot(range(Cd_test_ok.shape[0]),Cd_test_ok,label="Original Data")
    plt.legend(loc='best')
    plt.ylabel(r'$C_d$')
    plt.xlabel('Test samples')
    plt.show()
    plt.figure()
    plt.plot(range(Cd_test.shape[0]),pred_n,label="Predicted Data")
    plt.plot(range(Cd_test.shape[0]),Cd_test,label="Original Data")
    plt.legend(loc='best')
    plt.ylabel(r'$C_d$')
    plt.xlabel('Test samples')
    plt.show()
    