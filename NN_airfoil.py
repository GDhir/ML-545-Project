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
    num_epochs = number of epochs to train the NN
    """
    def __init__(self, x_dim = 16, y_dim = 1, learning_rate = 0.001, N_hlayers = 1
                 , n_neur = 10, num_epochs = 100):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_neur = n_neur
        self.learning_rate = learning_rate
        self.N_hlayers =  N_hlayers
        self.g = tf.Graph() #graph for TF
        self.num_epochs = num_epochs
        self.activation_function = tf.nn.relu #Activation function
        with self.g.as_default():
            self.model() #initilize model
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    
    def model(self):

        self.X = tf.placeholder(dtype=tf.float32, shape=[None,280], name='X')
        self.extra = tf.placeholder(dtype=tf.float32, shape=(None,2), name='extra')        
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.y_dim), name='y')
        x_1 = self.X[:,0:140]
        x_2 = tf.reverse(self.X[:,140:], [1])
        print(x_2.get_shape())
        x_input = tf.concat([x_1,x_2], 1)

        conv1 = tf.layers.conv2d(inputs=tf.reshape(x_input,[-1,2,140,1]), filters=4, kernel_size=[2,2], 
            padding='same', activation=self.activation_function)
        print(conv1.get_shape())
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size = [1,4], strides=[1,4])
        print(pool1.get_shape())
        conv2 = tf.layers.conv2d(inputs= pool1, filters=4, kernel_size=[2,2], 
            padding='same', activation=self.activation_function)
        print(conv2.get_shape())
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size = [2,7], strides=[2,7])
        print(pool2.get_shape())
        pool2_flat  = tf.reshape(pool2,[-1,5*4] )
        input_NN = tf.concat([pool2_flat, self.extra],1)
        #input layer
        layer_1 = tf.layers.dense(inputs=input_NN, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer1')
        
        d = {'layer_1':layer_1} #dictionary for the layer
        #Creation of the hidden layers
        for i in range(self.N_hlayers):
            d['layer_{0}'.format(i+2)] = tf.layers.dense(inputs=d['layer_{0}'.format(i+1)], units=self.n_neur, activation = self.activation_function, name = 'layer_{0}'.format(i+2))

        #last layer and actually the NN
        self.network = tf.layers.dense(inputs=d['layer_{0}'.format(self.N_hlayers+1)], units=self.y_dim, activation = None,
                                  name = 'layerf')

        #cost function to optimize
        self.cost = tf.reduce_mean(tf.square(self.network-self.y)) # our mean squared error cost function
        
        #optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def train_NN(self, X_train, y_train, X_test, y_test, y_train_ok, y_test_ok, verbosity = True, tolerance=1E-6,shuffle=False):
        """
        X_train: input X to train the NN
        y_train: normalized y_train 
        X_test: input X to evaluate the NN
        y_test: normalized y_test
        y_train_ok: y_train w/o normalization
        y_test_ok y_test w/o normalization
        shuffle: shuffle the train data during the optimization
        """ 
        self.sess = tf.Session(graph = self.g) #session for TF
        self.sess.run(self.init_op) 
        self.X_train = X_train 
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test
        self.y_train_ok = y_train_ok 
        self.y_test_ok = y_test_ok 
        self.shuffle  = shuffle
        self.tolerance = tolerance
        self.training_cost = [] #list to save the cost
        self.R_error_train = [] #list to save the relative error of the train set
        self.R_error_test = [] #lsit to save the relative error of the test set
        start = time.time() 
        #batch size used during the optimization
        self.batch_size = int(self.X_train.shape[0]*0.05)
        self.x_train_points = np.concatenate((self.X_train[:,0:140],self.X_train[:,141:281]),axis=1)
        self.feat_train_points = self.X_train[:,281:]
        self.x_test_points = np.concatenate((self.X_test[:,0:140],self.X_test[:,141:281]),axis=1)
        self.feat_test_points = self.X_test[:,281:] 
        # The optimization:
        for i in range(self.num_epochs):
            batch_generator = self.generate_batches()
            for batch_x, batch_e, batch_y in batch_generator:
                feed = {self.X: batch_x, self.extra: batch_e, self.y: batch_y}
                _ = self.sess.run([self.optimizer], feed_dict=feed)
            cost  = self.sess.run([self.cost], feed_dict={self.X: self.x_train_points, self.extra: self.feat_train_points, self.y: self.y_train})
            self.training_cost.append(cost)
            
            #evaluate the NN using x_test
            self.y_pred = self.sess.run(self.network, feed_dict={self.X: self.x_test_points, self.extra: self.feat_test_points})
            #denormalize the prediction
            self.y_pred_ok = self.denormalize(self.y_pred)
            #compute the relative error of the test data
            error_test = np.sqrt(np.sum((self.y_test_ok-self.y_pred_ok)**2))/np.sqrt(np.sum(self.y_test_ok**2))*100
            self.R_error_test.append(error_test)

            #evaluate the NN using x_train
            y_pred_train = self.sess.run(self.network, feed_dict={self.X: self.x_train_points, self.extra: self.feat_train_points})
            #denormalize the prediction
            y_pred_train_ok = self.denormalize(y_pred_train)
            #compute the relative error of the training data
            error_train = np.sqrt(np.sum((self.y_train_ok-y_pred_train_ok)**2))/np.sqrt(np.sum(self.y_train_ok**2))*100
            self.R_error_train.append(error_train)
            if verbosity:
                print('Epoch: ', i+1, 'Training cost: ', self.training_cost[-1])
            if np.asarray(self.training_cost[-1])<self.tolerance:
                break
        end = time.time()
        self.total_time = end-start
        print('====== NN trained, ',self.total_time/60, '[min] ======')

    def generate_batches(self):
        """
        Function to create  batches 
        """
        X_copy = np.array(self.X_train) 
        y_copy = np.array(self.y_train)
        if self.shuffle:
            data = np.column_stack((X_copy, y_copy))
            np.random.shuffle(data)
            X_copy = data[:,:-1]
            y_copy = data[:,-1].reshape((-1,1))
        for i in range(0, self.X_train.shape[0], self.batch_size):
            yield (np.concatenate((X_copy[i:i+self.batch_size,:140],X_copy[i:i+self.batch_size,141:281]),axis=1),X_copy[i:i+self.batch_size,281:], y_copy[i:i+self.batch_size])

    def generate_plot(self, name, show=False, save=False):
        """
        Function to create plots and save the NN
        name: name of the variable trained (Cd, Cm or Cl)
        show: show plots if True
        save: save NN, plots and a .txt file if True, using the 'name'
        """
        self.test_cost = self.sess.run(self.cost, feed_dict= {self.X: self.x_test_points, self.extra: self.feat_test_points, self.y: self.y_test})
    
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
        """
        Function to denormalize the output of the NN
        """
        y = self.y_train_ok.reshape(-1,1)
        y_normalized = y_normalized.reshape(-1,1)
        scl = MinMaxScaler()
        a = scl.fit_transform(y)
        new = scl.inverse_transform(y_normalized)
        return new

