from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

import numpy as np
import tensorflow as tf


def read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace('\n', ' <eos> ').split()
    

def file_to_tensor(filename, word_to_id):
    data = open(filename, 'r').read()
    return [word_to_id[word] for word in data]


class DataReader():
    
    def __init__(self, data, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        if not (os.path.exists(data)):
            raise IOError("Input data file not found.")
        else:
            self.build_vocab(data)
            
        self.generate_batches(self.get_tensor(data))
        
    def build_vocab(self, data):
        raw_data = open(data, 'r').read()
        
        counter = collections.Counter(raw_data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        self.tokens, _ = zip(*count_pairs)
        self.vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.vocab_size = len(self.tokens)
        
        
    def get_tensor(self, data):
        return file_to_tensor(data, self.vocab)
    

    def generate_batches(self, tensor):
        raw_tensor = np.array(tensor, dtype=np.int32)
        
        batch_len = len(raw_tensor) // (self.batch_size * self.seq_length)
        if batch_len == 0:
            raise ValueError("batch_len == 0, decrease batch_size or seq_length")
        
        self.num_batches = batch_len
        
        raw_tensor = raw_tensor[:self.batch_size * self.seq_length * batch_len]
        
        xdata = raw_tensor
        ydata = np.copy(raw_tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), batch_len, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), batch_len, 1)
        

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    
    def reset_batch_pointer(self):
        self.pointer = 0
        
