import numpy as np 
import tensorflow as tf 

class NeuralAirfoil(object):
    """
    x_dim = number of features
    y_dim = number of outputs 
    learning_rate = 'step' used by the optimizer
    n_neur = number of neurons per hidden layer
    N_hlayer = number of hidden layers
    """
    def __init__(self, x_dim = 16, y_dim = 1, learning_rate = 0.001, N_hlayers = 1
                 , n_neur = 10):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_neur = n_neur
        self.learning_rate = learning_rate
        self.N_hlayers =  N_hlayers
        self.g = tf.Graph()
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




