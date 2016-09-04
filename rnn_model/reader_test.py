from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

from reader import DataReader

class ReaderTest():
    
    def __init__(self):
        # Initialize fields
        self.batch_size = 3
        self.seq_length = 2

        string_data = "\n".join(
            [" hello there i am",
             " rain as day",
             " want some cheesy puffs ?"])
    
        # Create temporary data file
        tmpdir = tf.test.get_temp_dir()
        self.filename = os.path.join(tmpdir, "test.txt")
        with tf.gfile.GFile(self.filename, "w") as fh:
            fh.write(string_data)

        
    def testReader1(self):
        # Create DataReader instance
        data_reader = DataReader(self.filename, self.batch_size, self.seq_length)
        
        print (data_reader.vocab)
        print (data_reader.get_tensor(self.filename))

        
    def testReader2(self):
        # Create DataReader instance
        data_reader = DataReader(self.filename, self.batch_size, self.seq_length)
    
        tensor = data_reader.get_tensor(self.filename)
        
        data_reader.generate_batches(tensor)
        x = data_reader.x_batches
        y = data_reader.y_batches
        
        print(x)
        print(y)


if __name__ == "__main__":
    test = ReaderTest()
    test.testReader1()
    test.testReader2()
     
