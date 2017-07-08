from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from random import randint
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
# Import data
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

# Create the model
        x = tf.placeholder(tf.float32, [None, 2*784])
        W1 = tf.Variable(tf.random_normal([2*784, 256]))
        b1 = tf.Variable(tf.random_normal([256]))
        W2 = tf.Variable(tf.random_normal([256, 100]))
        b2 = tf.Variable(tf.random_normal([100]))
        y = tf.matmul(tf.matmul(x,W1) + b1, W2) + b2

# Define loss and optimizer
        y_ = tf.placeholder(tf.int64, [None])

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        train_data = mnist.train.images
        train_labels = mnist.train.labels

# Train
        print("Starting training")
        for round in range(6000):
                batch_xs = []
                batch_ys = []
                for _ in range(100):
                        a = randint(0,54999)
                        b = randint(0,54999)
                        batch_xs.append(np.concatenate([train_data[a],train_data[b]]))
                        batch_ys.append(train_labels[a] * 10 + train_labels[b])
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_data = mnist.test.images
        test_labels = mnist.test.labels

        testbatch_xs = []
        testbatch_ys = []
        print("Starting testing")
        for _ in range(len(test_data)):
                a = randint(0,9999)
                b = randint(0,9999)
                testbatch_xs.append(np.concatenate([test_data[a],test_data[b]]))
                testbatch_ys.append(test_labels[a] * 10 + test_labels[b])

        print(sess.run(accuracy, feed_dict={x: testbatch_xs, y_: testbatch_ys}))

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
        FLAGS, unparsed = parser.parse_known_args()
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
