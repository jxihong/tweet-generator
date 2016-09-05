from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

from reader import DataReader

class RNNModel():

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        
        size = config.hidden_size
        vocab_size = config.vocab_size
        
        self._input_data = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        
        #Define RNN tensor
        lstm_cell = rnn_cell.BasicLSTMCell(size)
        self.cells = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)        
        self._initial_state = cells.zero_state(self.config.batch_size, tf.float32)
        
        #Converting Input in an Embedded form
        with tf.device("/cpu:0"): #Tells Tensorflow what GPU to use specifically
            embedding = tf.get_variable("embedding", [vocab_size, size])
            embeddingLookedUp = tf.nn.embedding_lookup(embedding, self._input_data)
            inputs = tf.split(1, self.seq_size, embeddingLookedUp)
            inputTensorsAsList = [tf.squeeze(input_, [1]) for input_ in inputs]
        
        #Define softmax values
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])

        #Get hidden layer outputs
        hidden_layer_output, last_state = rnn.rnn(self.cells, inputTensorsAsList, initial_state=self._initial_state)
        hidden_layer_output = tf.reshape(tf.concat(1, hidden_layer_output), [-1, size])
        self._logits = tf.nn.xw_plus_b(hidden_layer_output, softmax_w, softmax_b)
        self._predictionSoftmax = tf.nn.softmax(self._logits)
        
        #Define the loss function
        loss = seq2seq.sequence_loss_by_example([self._logits], 
                                                [tf.reshape(self._targets, [-1])], 
                                                [tf.ones([self.batch_size * self.seq_length])], 
                                                vocab_size)
        self._cost = tf.div(tf.reduce_sum(loss), self.batch_size)
        self._final_state = last_state
        
        #Optimize gradient descent algorithm
        self._learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        
    def assign_learningRate(self, session, lr_value):
        session.run(tf.assign(self._learning_rate, lr_value))
            
    
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def train_op(self):
        return self._train_op
