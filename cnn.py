import numpy as np
import tensorflow as tf
from dataset import Dataset
from imgaug import augmenters as iaa

BATCH_SIZE = 32
IMG_SIZE = 96
EPOCHS = 200
LR = 1e-6
dataset = Dataset(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
NUM_CLASSES = dataset.num_classes

X = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='ground_truth')

W_conv1 = tf.get_variable('W_conv1', shape=[3, 3, 3, 32])
b_conv1 = tf.get_variable('b_conv1', shape=[32], initializer=tf.constant_initializer())
conv_1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
a_1 = tf.nn.relu(conv_1, name='relu_conv_1')

W_conv2 = tf.get_variable('W_conv2', shape=[3, 3, 32, 32])
b_conv2 = tf.get_variable('b_conv2', shape=[32], initializer=tf.constant_initializer())
conv_2 = tf.nn.conv2d(a_1, W_conv2, strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
a_2 = tf.nn.relu(conv_2, name='relu_conv_2')

pool_3 = tf.nn.max_pool(a_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_3')
drop_3 = tf.nn.dropout(pool_3, keep_prob=0.75, name='drop_3')

W_conv4 = tf.get_variable('W_conv4', shape=[3, 3, 32, 64])
b_conv4 = tf.get_variable('b_conv4', shape=[64], initializer=tf.constant_initializer())
conv_4 = tf.nn.conv2d(drop_3, W_conv4, strides=[1, 1, 1, 1], padding='SAME', name='conv_4')
a_4 = tf.nn.relu(conv_4, name='relu_conv_4')

W_conv5 = tf.get_variable('W_conv5', shape=[3, 3, 64, 64])
b_conv5 = tf.get_variable('b_conv5', shape=[32], initializer=tf.constant_initializer())
conv_5 = tf.nn.conv2d(a_4, W_conv5, strides=[1, 1, 1, 1], padding='VALID', name='conv_5')
a_5 = tf.nn.relu(conv_5, name='relu_conv_5')

pool_6 = tf.nn.max_pool(a_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_6')
drop_6 = tf.nn.dropout(pool_6, keep_prob=0.75, name='drop_6')

drop_6_shape = drop_6.get_shape().as_list()
flat_7 = tf.reshape(drop_6, shape=[-1, drop_6_shape[1]*drop_6_shape[2]*drop_6_shape[3]])
W_fc7 = tf.get_variable('W_fc7', shape=[drop_6_shape[1]*drop_6_shape[2]*drop_6_shape[3], 512])
b_fc7 = tf.get_variable('b_fc7', shape=[512], initializer=tf.constant_initializer())
h_fc7 = tf.nn.bias_add(tf.matmul(flat_7, W_fc7, name='fc_7_matmul'), b_fc7, name='fc_7_bias_add')
a_fc7 = tf.nn.relu(h_fc7, name='relu_fc_7')
drop_7 = tf.nn.dropout(a_fc7, keep_prob=0.5, name='drop_7')

W_out = tf.get_variable('W_out', shape=[512, NUM_CLASSES])
b_out = tf.get_variable('b_out', shape=[NUM_CLASSES], initializer=tf.constant_initializer())
h_out = tf.nn.bias_add(tf.matmul(drop_7, W_out, name='out_matmul'), b_out, name='out_bias_add')
y_ = tf.nn.softmax(h_out, name='output')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_out, labels=y))

optimizer = tf.train.RMSPropOptimizer(learning_rate=LR, momentum=0.9).minimize(mean_loss)

aug = iaa.SomeOf(2, [
    iaa.Affine(rotate=(-10, 10)),
    iaa.Fliplr(1),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AddElementwise((-40, 40)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.5*255), per_channel=0.5),
    iaa.CropAndPad(px=((0, 30), (0, 10), (0, 30), (0, 10)), pad_mode=ia.ALL, pad_cval=(0, 128))
])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_val, y_val = dataset.val_set
    # X_val = np.multiply(X_val, 1/255.)
    for epoch_i in range(EPOCHS):
        dataset.reset_batch_pointer()
        losses = []
        accuracies = []
        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
            X_batch, y_batch = dataset.next_batch
            X_batch = aug.augment_images(X_batch)
            # X_batch = np.multiply(X_batch, 1/255.)
            loss, _ = sess.run([mean_loss, optimizer], feed_dict={X: X_batch, y: y_batch})
            if batch_i%100 == 0:
                print('{}/{}, Epoch: {}, Cost: {}'.format(batch_num,
                                                        EPOCHS * dataset.num_batches_in_epoch(),
                                                        epoch_i, loss))
        for batch_num, batch_i in enumerate(range(y_val.shape[0]//BATCH_SIZE)):
            acc_val, loss_val = sess.run([accuracy, mean_loss], feed_dict={X: X_val[BATCH_SIZE*batch_num:BATCH_SIZE*batch_num+1], y: y_val[BATCH_SIZE*batch_num:BATCH_SIZE*batch_num+1]})
            accuracies.append(acc_val)
            losses.append(loss_val)
        losses = np.mean(losses)
        accuracies = np.mean(accuracies)
        print('Epoch: {}, Validation Cost: {}, Validation Accuracy: {}\n'.format(epoch_i, losses, accuracies))
