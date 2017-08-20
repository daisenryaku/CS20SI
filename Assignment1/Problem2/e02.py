import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

#learning_rate = 0.01
batch_size = 128
n_epochs = 100

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
increment_step = global_step.assign_add(1)

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")
logits = tf.matmul(X, w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	n_batches = int(MNIST.train.num_examples/batch_size)
	for i in range(n_epochs):
		increment_step = global_step.assign_add(1)
		learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
		sess.run([increment_step, learning_rate])
		for _ in range(n_batches):
			X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})

	n_batches = int(MNIST.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch, Y_batch = MNIST.test.next_batch(batch_size)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits],feed_dict={X: X_batch, Y:Y_batch})
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
		total_correct_preds += sess.run(accuracy)
	print("lr:{0}".format(sess.run(learning_rate)))
	print("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))
