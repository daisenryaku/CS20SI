import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
SKIP_WINDOW = 1
EMBED_SIZE = 128
NUM_SAMPLED = 64
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000

def word2vec(batch_gen):
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                            name='embed_matrix')

    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)),
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                            biases=nce_bias,
                                            labels=target_words,
                                            inputs=embed,
                                            num_sampled=NUM_SAMPLED,
                                            num_classes=VOCAB_SIZE), name='loss')

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            loss_batch, _ = sess.run([loss, optimizer],
                                    feed_dict={center_words: centers, target_words: targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()
