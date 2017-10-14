import numpy as np
import tensorflow as tf
from dataset import Dataset
from imgaug import augmenters as iaa

BATCH_SIZE = 128
IMG_SIZE = 96
EPOCHS = 50
dataset = Dataset(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
NUM_CLASSES = dataset.num_classes
LR = 0.01
NUM_EPOCHS_PER_DECAY = 1
DECAY_STEPS = dataset.num_batches_in_epoch() * NUM_EPOCHS_PER_DECAY
DECAY_RATE = 0.999999
MOMENTUM = 0.9

X = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='ground_truth')

W_conv1 = tf.get_variable('W_conv1', shape=[3, 3, 3, 32])
b_conv1 = tf.get_variable('b_conv1', shape=[32], initializer=tf.constant_initializer())
conv_1 = tf.nn.bias_add(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1'), b_conv1, name='conv_1_bias_add')
a_1 = tf.nn.relu(conv_1, name='relu_conv_1')

W_conv2 = tf.get_variable('W_conv2', shape=[3, 3, 32, 32])
b_conv2 = tf.get_variable('b_conv2', shape=[32], initializer=tf.constant_initializer())
conv_2 = tf.nn.bias_add(tf.nn.conv2d(a_1, W_conv2, strides=[1, 1, 1, 1], padding='VALID', name='conv_2'), b_conv2, name='conv_2_bias_add')
a_2 = tf.nn.relu(conv_2, name='relu_conv_2')

pool_3 = tf.nn.max_pool(a_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_3')
drop_3 = tf.nn.dropout(pool_3, keep_prob=0.80, name='drop_3')

W_conv4 = tf.get_variable('W_conv4', shape=[3, 3, 32, 64])
b_conv4 = tf.get_variable('b_conv4', shape=[64], initializer=tf.constant_initializer())
conv_4 = tf.nn.bias_add(tf.nn.conv2d(drop_3, W_conv4, strides=[1, 1, 1, 1], padding='SAME', name='conv_4'), b_conv4, name='conv_4_bias_add')
a_4 = tf.nn.relu(conv_4, name='relu_conv_4')

W_conv5 = tf.get_variable('W_conv5', shape=[3, 3, 64, 64])
b_conv5 = tf.get_variable('b_conv5', shape=[64], initializer=tf.constant_initializer())
conv_5 = tf.nn.bias_add(tf.nn.conv2d(a_4, W_conv5, strides=[1, 1, 1, 1], padding='VALID', name='conv_5'), b_conv5, name='conv_5_bias_add')
a_5 = tf.nn.relu(conv_5, name='relu_conv_5')

pool_6 = tf.nn.max_pool(a_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_6')
drop_6 = tf.nn.dropout(pool_6, keep_prob=0.80, name='drop_6')

W_conv7 = tf.get_variable('W_conv7', shape=[3, 3, 64, 256])
b_conv7 = tf.get_variable('b_conv7', shape=[256], initializer=tf.constant_initializer())
conv_7 = tf.nn.bias_add(tf.nn.conv2d(drop_6, W_conv7, strides=[1, 1, 1, 1], padding='VALID', name='conv_7'), b_conv7, name='conv_7_bias_add')
a_7 = tf.nn.relu(conv_7, name='relu_conv_7')

W_conv8 = tf.get_variable('W_conv8', shape=[3, 3, 256, 256])
b_conv8 = tf.get_variable('b_conv8', shape=[256], initializer=tf.constant_initializer())
conv_8 = tf.nn.bias_add(tf.nn.conv2d(a_7, W_conv8, strides=[1, 1, 1, 1], padding='VALID', name='conv_8'), b_conv8, name='conv_8_bias_add')
a_8 = tf.nn.relu(conv_8, name='relu_conv_8')

pool_9 = tf.nn.max_pool(a_8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_9')
drop_9 = tf.nn.dropout(pool_9, keep_prob=0.80, name='drop_9')

drop_9_shape = drop_9.get_shape().as_list()
flat_10 = tf.reshape(drop_9, shape=[-1, drop_9_shape[1]*drop_9_shape[2]*drop_9_shape[3]])
W_fc10 = tf.get_variable('W_fc10', shape=[drop_9_shape[1]*drop_9_shape[2]*drop_9_shape[3], 1024])
b_fc10 = tf.get_variable('b_fc10', shape=[1024], initializer=tf.constant_initializer())
h_fc10 = tf.nn.bias_add(tf.matmul(flat_10, W_fc10, name='fc_10_matmul'), b_fc10, name='fc_10_bias_add')
a_fc10 = tf.nn.relu(h_fc10, name='relu_fc_10')
drop_10 = tf.nn.dropout(a_fc10, keep_prob=0.5, name='drop_10')

W_fc11 = tf.get_variable('W_fc11', shape=[1024, 1024])
b_fc11 = tf.get_variable('b_fc11', shape=[1024], initializer=tf.constant_initializer())
h_fc11 = tf.nn.bias_add(tf.matmul(drop_10, W_fc11, name='fc_11_matmul'), b_fc11, name='fc_11_bias_add')
a_fc11 = tf.nn.relu(h_fc11, name='relu_fc_11')
drop_11 = tf.nn.dropout(a_fc11, keep_prob=0.5, name='drop_11')

W_out = tf.get_variable('W_out', shape=[1024, NUM_CLASSES])
b_out = tf.get_variable('b_out', shape=[NUM_CLASSES], initializer=tf.constant_initializer())
h_out = tf.nn.bias_add(tf.matmul(drop_11, W_out, name='out_matmul'), b_out, name='out_bias_add')
y_ = tf.nn.softmax(h_out, name='output')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_out, labels=y))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=LR, global_step=global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM, use_nesterov=True).minimize(mean_loss, global_step=global_step)

aug = iaa.SomeOf(4, [
    iaa.Affine(rotate=(-30, 30)),
    iaa.Fliplr(1),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AddElementwise((-40, 40)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.5*255), per_channel=0.5)
])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_val, y_val = dataset.val_set
    X_val = np.multiply(X_val, 1/255.)
    for epoch_i in range(EPOCHS):
        dataset.reset_batch_pointer()
        losses = []
        accuracies = []
        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
            X_batch, y_batch = dataset.next_batch
            X_batch = aug.augment_images(X_batch)
            X_batch = np.multiply(X_batch, 1/255.)
            loss, _, acc_train = sess.run([mean_loss, optimizer, accuracy], feed_dict={X: X_batch, y: y_batch})
            if batch_num%25 == 0:
                print('{}/{}, Epoch: {}, Cost: {}, Accuracy: {}'.format(batch_num,
                                                                        EPOCHS * dataset.num_batches_in_epoch(),
                                                                        epoch_i, loss, acc_train))
        for batch_num, batch_i in enumerate(range(y_val.shape[0]//BATCH_SIZE)):
            acc_val, loss_val = sess.run([accuracy, mean_loss], feed_dict={X: X_val[BATCH_SIZE*batch_num:BATCH_SIZE*batch_num+1], y: y_val[BATCH_SIZE*batch_num:BATCH_SIZE*batch_num+1]})
            accuracies.append(acc_val)
            losses.append(loss_val)
        losses = np.mean(losses)
        accuracies = np.mean(accuracies)
        print('Epoch: {}, Validation Cost: {}, Validation Accuracy: {}\n'.format(epoch_i, losses, accuracies))
