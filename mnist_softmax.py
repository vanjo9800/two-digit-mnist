from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from random import shuffle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

# Create the model
	x = tf.placeholder(tf.float32, [None, 2*784])
	W1 = tf.Variable(tf.zeros([2*784, 50]))
	b1 = tf.Variable(tf.zeros([50]))
	W2 = tf.Variable(tf.zeros([50, 100]))
	b2 = tf.Variable(tf.zeros([100]))
	y = tf.matmul(tf.nn.relu(tf.matmul(x,W1) + b1), W2) + b2

# Define loss and optimizer
	y_ = tf.placeholder(tf.int64, [None])

	cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	train_data = mnist.train.images
	train_labels = mnist.train.labels
	train_numbers = list(range(len(train_data)))
	train_numbers += list(range(len(train_data)))
	train_numbers += list(range(len(train_data)))
	train_numbers += list(range(len(train_data)))
	shuffle(train_numbers)

# Train
	print("Starting training")
	step = 0
	for round in range(1000):
		batch_xs = []
		batch_ys = []
		for _ in range(100):
			batch_xs.append(np.concatenate([train_data[train_numbers[step]],train_data[train_numbers[step+1]]]))
			batch_ys.append(train_labels[train_numbers[step]] * 10 + train_labels[train_numbers[step+1]])
			step+=2
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	test_data = mnist.test.images
	test_labels = mnist.test.labels
	test_numbers = list(range(len(test_data)))
	test_numbers += list(range(len(test_data)))
	shuffle(test_numbers)

	testbatch_xs = []
	testbatch_ys = []
	step = 0
	print("Starting testing")
	for _ in range(len(test_data)):
		testbatch_xs.append(np.concatenate([test_data[test_numbers[step]],test_data[test_numbers[step+1]]]))
		testbatch_ys.append(test_labels[test_numbers[step]] * 10 + test_labels[test_numbers[step+1]])
		step+=2

	print(sess.run(accuracy, feed_dict={x: testbatch_xs, y_: testbatch_ys}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
