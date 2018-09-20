from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_x, batch_y = mnist.train.next_batch(1000)
print(batch_x.shape[0],batch_x.shape[1])

print(batch_y.shape[0],batch_y.shape[1])

batch_x = batch_x.reshape((1000, 28, 28))

print(batch_x.shape[0],batch_x.shape[1],batch_x.shape[2])