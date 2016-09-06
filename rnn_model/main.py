from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reader import DataReader
from rnn import RNNModel

import argparse
import time
import os, sys


class ParameterConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    seq_length = 35
    batch_size = 20
    hidden_size = 800
    max_epoch = 10
    total_max_epoch = 50
    keep_prob = 0.5
    lr_decay = 0.97


def main(args):
    cmd = args.get('cmd')
    
    if cmd == 'train':
        train(args)
    else:
        print("Invalid Command: %s" % cmd)


def train(args):
    config = ParameterConfig()

    data_reader = DataReader(args['data'], config.batch_size, config.seq_length)
    config.vocab_size = data_reader.vocab_size

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            training_model = RNNModel(config=config)
            
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        
        #Run a single epoch of training
        for epoch in range(config.total_max_epoch):
            current_state = session.run(training_model.initial_state)
                    
            learning_rate_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            training_model.assign_learningRate(session, config.learning_rate * learning_rate_decay)
                    
            total_cost = 0.0
            total_seq = 0
                    
            data_reader.reset_batch_pointer()
            for batch in range(data_reader.num_batches):
                start = time.time()
                x,y = data_reader.next_batch()
                feed_dict = {training_model.input_data: x, training_model.targets: y, 
                             training_model.initial_state: current_state}
                  
                cost, current_state, _ = session.run([training_model.cost, training_model.final_state, training_model.train_op], feed_dict) 
                 
                total_cost += cost
                total_seq += config.seq_length
                 
                perplexity = np.exp(total_cost / total_seq)
                end = time.time()                 
                
                print("{}/{} (epoch {}), perplexity = {:.3f}, time/batch = {:.3f}" \
                    .format(epoch * data_reader.num_batches + batch,
                            config.total_max_epoch * data_reader.num_batches,
                            epoch, perplexity, end - start))
                
                if (epoch == config.total_max_epoch - 1 and batch == data_reader.num_batches - 1):
                    checkpoint_path = os.path.join(args['model_dir'], 'model.ckpt')
                    saver.save(session, checkpoint_path, global_step = epoch * data_reader.num_batches + batch)
                    print("model saved to {}".format(checkpoint_path))


    session.close()

                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(help ='sub-commands', dest='cmd')
    train_parser = subparsers.add_parser('train', help='train neural network')
    train_parser.add_argument('--data', type=str, required=True, help='data file containing training samples')
    train_parser.add_argument('--model_dir', type=str, required=True, help='directory to store checkpoint models')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    main(args)
