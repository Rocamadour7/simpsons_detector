import tensorflow as tf
import numpy as np
from dataset import Dataset

NUM_CLASSES = 20
NUM_EPOCHS = 50
BATCH_SIZE = 256
LR = 5e-4
IMG_SIZE = 99
NUM_CHANNELS = 3

dataset = Dataset(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

with tf.name_scope('ConvLayer_1'):
    W_conv1 = tf.get_variable('W_conv1', shape=[11, 11, NUM_CHANNELS, 96])
    b_conv1 = tf.get_variable('b_conv1', shape=[96])

    a1 = tf.nn.conv2d(X, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1
    pool_a1 = tf.nn.max_pool(a1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('ConvLayer_2'):
    W_conv2 = tf.get_variable('W_conv2', shape=[5, 5, 96, 256])
    b_conv2 = tf.get_variable('b_conv2', shape=[256])

    a2 = tf.nn.conv2d(pool_a1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
    pool_a2 = tf.nn.max_pool(a2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('ConvLayer_3'):
    W_conv3 = tf.get_variable('W_conv3', shape=[3, 3, 256, 384])
    b_conv3 = tf.get_variable('b_conv3', shape=[384])

    a3 = tf.nn.conv2d(pool_a2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3

with tf.name_scope('ConvLayer_4'):
    W_conv4 = tf.get_variable('W_conv4', shape=[3, 3, 384, 384])
    b_conv4 = tf.get_variable('b_conv4', shape=[384])

    a4 = tf.nn.conv2d(a3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4

with tf.name_scope('ConvLayer_5'):
    W_conv5 = tf.get_variable('W_conv5', shape=[3, 3, 384, 256])
    b_conv5 = tf.get_variable('b_conv5', shape=[256])

    a5 = tf.nn.conv2d(a4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5
    pool_a5 = tf.nn.max_pool(a5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('FullyConnectedLayer_6'):
    pool_a5_shape = pool_a5.get_shape().as_list()
    h_reshaped = tf.reshape(pool_a5, shape=[-1, pool_a5_shape[1]*pool_a5_shape[2]*pool_a5_shape[3]])

    W_fc6 = tf.get_variable('W_fc6', shape=[pool_a5_shape[1]*pool_a5_shape[2]*pool_a5_shape[3], 4096])
    b_fc6 = tf.get_variable('b_fc6', shape=[4096])

    a6 = tf.nn.bias_add(tf.matmul(h_reshaped, W_fc6), b_fc6)
    h_fc6 = tf.nn.relu(a6)

with tf.name_scope('FullyConnectedLayer_7'):
    W_fc7 = tf.get_variable('W_fc7', shape=[4096, 4096])
    b_fc7 = tf.get_variable('b_fc7', shape=[4096])

    a7 = tf.nn.bias_add(tf.matmul(h_fc6, W_fc7), b_fc7)
    h_fc7 = tf.nn.relu(a7)

with tf.name_scope('FullyConnectedLayer_8'):
    W_fc8 = tf.get_variable('W_fc8', shape=[4096, NUM_CLASSES])
    b_fc8 = tf.get_variable('b_fc8', shape=[NUM_CLASSES])

    y_ = tf.nn.bias_add(tf.matmul(h_fc7, W_fc8), b_fc8)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('Loss'):
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(LR).minimize(mean_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_val, y_val = dataset.val_set
    for epoch_i in range(NUM_EPOCHS):
        dataset.reset_batch_pointer()
        losses = []
        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
            X_batch, y_batch = dataset.next_batch
            loss, _ = sess.run([mean_loss, optimizer], feed_dict={X: X_batch, y: y_batch})
            losses.append(loss)
            print('{}/{}, Epoch: {}, Cost: {}'.format(batch_num,
                                                      NUM_EPOCHS * dataset.num_batches_in_epoch(),
                                                      epoch_i, loss))
        acc = sess.run(accuracy, feed_dict={X: X_val, y: y_val})
        losses = np.mean(losses)
        print('===========Epoch: {}, Cost: {}, Accuracy: {}==========='.format(epoch_i, losses, acc))
