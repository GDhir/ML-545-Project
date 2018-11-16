import numpy as np
import tensorflow as tf
from functions import read_data, neural_net_model

X_train, y_train, X_test, y_test, df_test, df_train= read_data()

xs = tf.placeholder("float")
xd = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net_model(xs,3)
cost = tf.reduce_mean(tf.square(output-ys))
# our mean squared error cost function

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

c_t =[]
saver = tf.train.Saver()
# Gradinent Descenwith tf.Session() as sess:
with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        for j in range(X_train.shape[0]):
            sess.run(train,feed_dict= {xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
            # Run train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        print('Epoch :',i,'Cost :',c_t[i])
    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    save_path = saver.save(sess,"./NN_ex.ckpt")   

    
       

