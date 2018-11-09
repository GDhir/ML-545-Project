import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import read_data, neural_net_model_1, denormalize

X_train, y_train_n, X_test, y_test_n,  y_train, y_test= read_data()
Cd_test_ok = y_test[:,2]
xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model_1(xs,16)
cost = tf.reduce_mean(tf.square(output-ys))
saver = tf.train.Saver()
with tf.Session() as session:
    #Load NN
    saver.restore(session, './data/NN_Cd_1.ckpt')
    # Evalute test data
    pred = session.run(output,feed_dict={xs:X_test})
    Mse1 = np.mean((pred-y_test_n[:,2])**2)
    Mse2 = session.run(cost, feed_dict={xs:X_test,ys:y_test_n[:,2]})
    print(Mse1,Mse2)
    pred = denormalize(Cd_test_ok,pred)
    
    Mse = np.mean((pred-Cd_test_ok)**2)
    print(Mse)
    #Plot results
    plt.plot(range(Cd_test_ok.shape[0]),Cd_test_ok,label="Original Data")
    plt.plot(range(Cd_test_ok.shape[0]),pred,label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel(r'$C_d$')
    plt.xlabel('Test samples')
    plt.show()
    