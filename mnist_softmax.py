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
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Create the model
	x = tf.placeholder(tf.float32, [None, 2*784])
	W = tf.Variable(tf.zeros([2*784, 100]))
	b = tf.Variable(tf.zeros([100]))
	y = tf.matmul(x, W) + b

# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 100])

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.7).minimize(cross_entropy)

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
			current_label = np.zeros(100)
			current_label[list(train_labels[train_numbers[step]]).index(1) * 10 + list(train_labels[train_numbers[step+1]]).index(1)] = 1
			batch_ys.append(current_label)
			step+=2
		#batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
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
		current_label = np.zeros(100)
		current_label[list(test_labels[test_numbers[step]]).index(1) * 10 + list(test_labels[test_numbers[step+1]]).index(1)] = 1
		testbatch_ys.append(current_label)
		step+=2

	print(sess.run(accuracy, feed_dict={x: testbatch_xs, y_: testbatch_ys}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
