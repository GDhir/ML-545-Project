import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functions import read_data, neural_net_model, denormalize

_,_,X_test, y_test, df_test,_ = read_data()

xs = tf.placeholder("float")
output = neural_net_model(xs,3)
saver = tf.train.Saver()
with tf.Session() as session:
    #Load NN
    saver.restore(session, './NN_ex.ckpt')
    # Evalute test data
    pred = session.run(output,feed_dict={xs:X_test})
    y_test = denormalize(df_test,y_test)
    pred = denormalize(df_test,pred)
    #Plot results
    plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('Stock Value')
    plt.xlabel('Days')
    plt.title('Stock Market Nifty')
    plt.show()

