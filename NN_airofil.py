import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler


class NeuralAirfoil(object):
    """
    x_dim = number of features
    y_dim = number of outputs 
    learning_rate = 'step' used by the optimizer
    n_neur = number of neurons per hidden layer
    N_hlayer = number of hidden layers
    """
    def __init__(self, x_dim = 16, y_dim = 1, learning_rate = 0.001, N_hlayers = 1
                 , n_neur = 10, num_epochs = 100):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_neur = n_neur
        self.learning_rate = learning_rate
        self.N_hlayers =  N_hlayers
        self.g = tf.Graph()
        self.num_epochs = num_epochs
        self.activation_function = tf.nn.relu
        with self.g.as_default():
            self.model()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    
    def model(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None,self.x_dim), name='X')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.y_dim), name='y')
        
        layer_1 = tf.layers.dense(inputs=self.X, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer1')
        d = {'layer_1':layer_1}
        for i in range(self.N_hlayers):
            d['layer_{0}'.format(i+2)] = tf.layers.dense(inputs=d['layer_{0}'.format(i+1)], units=self.n_neur, activation = self.activation_function, name = 'layer_{0}'.format(i+2))

        self.network = tf.layers.dense(inputs=d['layer_{0}'.format(self.N_hlayers+1)], units=self.y_dim, activation = None,
                                  name = 'layerf')

        self.cost = tf.reduce_mean(tf.square(self.network-self.y)) # our mean squared error cost function
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def train_NN(self, X_train, y_train, X_test, y_test, y_train_ok, y_test_ok, shuffle=False):
        self.sess = tf.Session(graph = self.g)
        self.sess.run(self.init_op)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test 
        self.y_train_ok = y_train_ok
        self.y_test_ok = y_test_ok
        self.shuffle  = shuffle
        self.training_cost = []
        self.R_error_train = []
        self.R_error_test = []
        start = time.time()
        self.batch_size = int(self.X_train.shape[0]*0.05)
        for i in range(self.num_epochs):
            batch_generator = self.generate_batches()
            for batch_x, batch_y in batch_generator:
                feed = {self.X: batch_x, self.y: batch_y}
                _, cost = self.sess.run([self.optimizer, self.cost], feed_dict=feed)
            
            self.training_cost.append(cost)
            
            self.y_pred = self.sess.run(self.network, feed_dict={self.X: self.X_test})
            self.y_pred_ok = self.denormalize(self.y_pred)
            error_test = np.sqrt(np.sum((self.y_test_ok-self.y_pred_ok)**2))/np.sqrt(np.sum(self.y_test_ok**2))*100
            self.R_error_test.append(error_test)

            y_pred_train = self.sess.run(self.network, feed_dict={self.X: self.X_train})
            y_pred_train_ok = self.denormalize(y_pred_train)
            error_train = np.sqrt(np.sum((self.y_train_ok-y_pred_train_ok)**2))/np.sqrt(np.sum(self.y_train_ok**2))*100
            self.R_error_train.append(error_train)
            
            print('Epoch: ', i+1, 'Training cost: ', self.training_cost[-1])
        
        end = time.time()
        self.total_time = end-start
        print('====== NN trained, ',self.total_time/60, '[min] ======')

    def generate_batches(self):
        X_copy = np.array(self.X_train) 
        y_copy = np.array(self.y_train)
        if self.shuffle:
            data = np.column_stack((X_copy, y_copy))
            np.random.shuffle(data)
            X_copy = data[:,:-1]
            y_copy = data[:,-1].reshape((-1,1))
        for i in range(0, self.X_train.shape[0], self.batch_size):
            yield (X_copy[i:i+self.batch_size,:], y_copy[i:i+self.batch_size])

    def generate_plot(self, name, show=False, save=False):
        self.test_cost = self.sess.run(self.cost, feed_dict= {self.X:self.X_test, self.y: self.y_test})
    
        fig1 = plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.semilogy(range(len(self.training_cost)),self.training_cost)
        plt.xlabel('Epoch')
        plt.ylabel('Training Cost')
        plt.subplot(1,2,2)
        plt.plot(range(len(self.training_cost)),self.training_cost)
        plt.xlabel('Epoch')

        fig2 = plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.plot(range(len(self.y_test)), self.y_test, label= name +' Test')
        plt.plot(range(len(self.y_test)), self.y_pred, label=name + ' Predicted')
        plt.xlabel('Test Data')
        plt.ylabel(name)
        plt.title('Normalized data')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(range(len(self.y_test)), self.y_test_ok, label= name +' Test')
        plt.plot(range(len(self.y_test)), self.y_pred_ok, label=name + ' Predicted')
        plt.xlabel('Test Data')
        plt.title('Actual data')
        plt.legend()
            
        fig3 = plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.plot(range(len(self.y_test)), self.y_test-self.y_pred)
        plt.xlabel('Test Data')
        plt.ylabel('Error '+name)
        plt.title('Normalized data')
        plt.subplot(1,2,2)
        plt.plot(range(len(self.y_test)), self.y_test_ok-self.y_pred_ok)
        plt.xlabel('Test Data')
        plt.title('Actual data')

        fig4 = plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.semilogy(range(len(self.R_error_test)), self.R_error_test, label='Test')
        plt.plot(range(len(self.R_error_train)), self.R_error_train, label='Train')        
        plt.xlabel('Epoch')
        plt.ylabel('Relative Error %'+name)
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(range(len(self.R_error_test)), self.R_error_test, label='Test')
        plt.plot(range(len(self.R_error_train)), self.R_error_train, label='Train')        
        plt.xlabel('Epoch')
        plt.legend()

        if show:
            plt.show()
        
        if save:
            save_path = self.saver.save(self.sess,"./data/NN_"+name+".ckpt")

            fig1.savefig('./data/training_cost'+name+'.pdf',bbox_inches='tight')
            fig2.savefig('./data/compare'+name+'.pdf',bbox_inches='tight')
            fig3.savefig('./data/error'+name+'.pdf',bbox_inches='tight')
            fig4.savefig('./data/relativeerror'+name+'.pdf',bbox_inches='tight')

            f = open("./data/"+name,"w+")
            f.write("%5.8f \n" %self.learning_rate)
            f.write("%5.8f \n" %self.n_neur)
            f.write("%5.8f \n" %self.N_hlayers)
            f.write("%5.8f \n"% self.test_cost)
            f.write("%5.8f \n"%self.total_time)
            f.write("%5.8f \n"%self.R_error_test[-1])
            f.write("%5.8f \n"%self.R_error_train[-1])
            for item in self.training_cost:
                f.write("%s " % item)
            f.close()

    def denormalize(self, y_normalized):
        y = self.y_train_ok.reshape(-1,1)
        y_normalized = y_normalized.reshape(-1,1)
        scl = MinMaxScaler()
        a = scl.fit_transform(y)
        new = scl.inverse_transform(y_normalized)
        return new

