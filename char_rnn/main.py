from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reader import DataReader
from rnn import RNNModel
from six.moves import cPickle

import argparse
import time
import os, sys

class ParameterConfig(object):
    init_scale = 0.05
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 2
    seq_length = 35
    batch_size = 50
    hidden_size = 1000
    max_epoch = 15
    total_max_epoch = 1000
    keep_prob = 0.75
    lr_decay = 0.97


def main(args):
    cmd = args.get('cmd')
    
    if cmd == 'train':
        train(args)
    elif cmd == 'predict':
        predict(args)
    else:
        print("Invalid Command: %s" % cmd)


def train(args):
    config = ParameterConfig()

    data_reader = DataReader(args['data'], config.batch_size, config.seq_length)
    config.vocab_size = data_reader.vocab_size
    
    if not os.path.exists(args['model_dir']):
        os.makedirs(args['model_dir'])
        
    with open(os.path.join(args['model_dir'], 'config.pkl'), 'wb') as f:
        cPickle.dump(config, f)
    with open(os.path.join(args['model_dir'], 'vocab.pkl'), 'wb') as f:
        cPickle.dump((data_reader.tokens, data_reader.vocab), f)

    training_model = RNNModel(config=config)
            
    with tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
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
                  
                cost, current_state, _ = session.run([training_model.cost, training_model.final_state, training_model.train_op], 
                                                     feed_dict) 
                 
                total_cost += cost
                total_seq += config.seq_length
                 
                perplexity = np.exp(total_cost / total_seq)
                end = time.time()                 
                
                print("{}/{} (epoch {}), perplexity = {:.3f}, time/batch = {:.3f}" \
                    .format(epoch * data_reader.num_batches + batch,
                            config.total_max_epoch * data_reader.num_batches,
                            epoch, perplexity, end - start))
                sys.stdout.flush()

                if ((epoch * data_reader.num_batches + batch) % 1000 == 0 \
                        or (epoch == config.total_max_epoch - 1 and batch == data_reader.num_batches - 1)):
                    
                    checkpoint_path = os.path.join(args['model_dir'], 'model.ckpt')
                    saver.save(session, checkpoint_path, global_step = epoch * data_reader.num_batches + batch)
                    print("Model saved to {}".format(checkpoint_path))
                    sys.stdout.flush()

    session.close()


def pick(probs):
    t = np.cumsum(probs)
    s = np.sum(probs)

    return(int(np.searchsorted(t, np.random.rand(1) * s)))


def predict(args):
    if not os.path.exists(args['model_dir']):
        raise IOError("Model directory doesn't exist: %s" %(args['model_dir']))

    with open(os.path.join(args['model_dir'], 'config.pkl'), 'rb') as f:
        config = cPickle.load(f)
    with open(os.path.join(args['model_dir'], 'vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
        
    config.batch_size = 1
    config.seq_length = 1
    prediction_model = RNNModel(config=config)
    
    with tf.Session() as session:
        tf.initialize_all_variables().run()
        
        if not os.path.exists(args['model']):
            raise IOError("Model file doesn't exist: %s" %(args['model']))
            
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(session, args['model'])

        state = session.run(prediction_model.cells.zero_state(1, tf.float32))    
        
        output = args['prime']
        for i in range(args['num_chars']):
            char = output[i]
            x = np.full((config.batch_size, config.seq_length), vocab[char], dtype=np.int32)
            feed = {prediction_model.input_data: x, prediction_model.initial_state: state}
                
            [predictionSoftmax, state] =  session.run([prediction_model._predictionSoftmax, prediction_model.final_state], 
                                                      feed)
            probs = predictionSoftmax[0]
            
            next_char = chars[pick(probs)]
            output += next_char
            char = next_char
                
        print('Prediction: %s \n' % (output))
        sys.stdout.flush()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(help ='sub-commands', dest='cmd')
    train_parser = subparsers.add_parser('train', help='train neural network')
    train_parser.add_argument('--data', type=str, required=True, help='data file containing training samples')
    train_parser.add_argument('--model_dir', type=str, required=True, help='directory to store checkpoint models')
    
    predict_parser = subparsers.add_parser('predict', help='predict using neural network')
    predict_parser.add_argument('--model_dir', type=str, required=True, help='model directory')
    predict_parser.add_argument('--model', type=str, required=True, help='model file')
    predict_parser.add_argument('-n', '--num_chars', type=int, default=500, help='number of characters to sample')
    predict_parser.add_argument('--prime', type=str, default=u'The ', help='prime text')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    main(args)
