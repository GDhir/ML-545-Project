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
                 , n_neur = 10, random_seed = None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_neur = n_neur
        self.learning_rate = learning_rate
        self.N_hlayers =  N_hlayers
        self.g = tf.Graph()
        self.activation_function = tf.nn.relu
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.model()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    
    def model(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None,self.x_dim), name='X')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.y_dim), name='y')
        
        layer_1 = tf.layers.dense(inputs=self.X, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer1')

        if self.N_hlayers == 1:
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer2')
            layer_f = layer_2
        elif self.N_hlayers == 2:
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer2')
            layer_3 = tf.layers.dense(inputs=layer_2, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer3')
            layer_f = layer_3
        elif self.N_hlayers == 3:
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer2')
            layer_3 = tf.layers.dense(inputs=layer_2, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer3')
            layer_4 = tf.layers.dense(inputs=layer_3, units=self.n_neur, activation = self.activation_function,
                                  name = 'layer4')
            layer_f = layer_4
 
        self.network = tf.layers.dense(inputs=layer_f, units=self.y_dim, activation = None,
                                  name = 'layerf')

        self.cost = tf.reduce_mean(tf.square(self.network-self.y)) # our mean squared error cost function
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)




