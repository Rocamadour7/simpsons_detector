import numpy as np
import tensorflow as tf
from dataset import Dataset
from imgaug import augmenters as iaa
import os

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

SAVE_FOLDER = 'save/'
LOG_FOLDER = 'logs/'

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

X = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='ground_truth')

drop_conv = tf.placeholder(tf.float32)
drop_fc = tf.placeholder(tf.float32)

with tf.name_scope('Conv_1'):
    with tf.name_scope('Weights'):
        W_conv1 = tf.get_variable('W_conv1', shape=[3, 3, 3, 32])
        variable_summary(W_conv1)
    with tf.name_scope('Biases'):
        b_conv1 = tf.get_variable('b_conv1', shape=[32], initializer=tf.constant_initializer())
        variable_summary(b_conv1)
    with tf.name_scope('Wx_plus_b'):
        conv_1 = tf.nn.bias_add(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1'), b_conv1, name='conv_1_bias_add')
        tf.summary.histogram('Pre_activations', conv_1)
    a_1 = tf.nn.relu(conv_1, name='relu_conv_1')
    tf.summary.histogram('Activation', a_1)

with tf.name_scope('Conv_2'):
    with tf.name_scope('Weights'):
        W_conv2 = tf.get_variable('W_conv2', shape=[3, 3, 32, 32])
        variable_summary(W_conv2)
    with tf.name_scope('Biases'):
        b_conv2 = tf.get_variable('b_conv2', shape=[32], initializer=tf.constant_initializer())
        variable_summary(b_conv2)
    with tf.name_scope('Wx_plus_b'):
        conv_2 = tf.nn.bias_add(tf.nn.conv2d(a_1, W_conv2, strides=[1, 1, 1, 1], padding='VALID', name='conv_2'), b_conv2, name='conv_2_bias_add')
        tf.summary.histogram('Pre_activations', conv_2)
    a_2 = tf.nn.relu(conv_2, name='relu_conv_2')
    tf.summary.histogram('Activation', a_2)
    pool_2 = tf.nn.max_pool(a_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')
    tf.summary.histogram('MaxPooling', pool_2)
    drop_2 = tf.nn.dropout(pool_2, keep_prob=drop_conv, name='drop_2')

with tf.name_scope('Conv_3'):
    with tf.name_scope('Weights'):
        W_conv3 = tf.get_variable('W_conv3', shape=[3, 3, 32, 64])
        variable_summary(W_conv3)
    with tf.name_scope('Biases'):
        b_conv3 = tf.get_variable('b_conv3', shape=[64], initializer=tf.constant_initializer())
        variable_summary(b_conv3)
    with tf.name_scope('Wx_plus_b'):
        conv_3 = tf.nn.bias_add(tf.nn.conv2d(drop_2, W_conv3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3'), b_conv3, name='conv_3_bias_add')
        tf.summary.histogram('Pre_activations', conv_2)
    a_3 = tf.nn.relu(conv_3, name='relu_conv_3')
    tf.summary.histogram('Activation', a_3)

with tf.name_scope('Conv_4'):
    with tf.name_scope('Weights'):
        W_conv4 = tf.get_variable('W_conv4', shape=[3, 3, 64, 64])
        variable_summary(W_conv4)
    with tf.name_scope('Biases'):
        b_conv4 = tf.get_variable('b_conv4', shape=[64], initializer=tf.constant_initializer())
        variable_summary(b_conv4)
    with tf.name_scope('Wx_plus_b'):
        conv_4 = tf.nn.bias_add(tf.nn.conv2d(a_3, W_conv4, strides=[1, 1, 1, 1], padding='VALID', name='conv_4'), b_conv4, name='conv_4_bias_add')
        tf.summary.histogram('Pre_activations', conv_4)
    a_4 = tf.nn.relu(conv_4, name='relu_conv_4')
    tf.summary.histogram('Activation', a_4)
    pool_4 = tf.nn.max_pool(a_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_4')
    tf.summary.histogram('MaxPooling', pool_4)
    drop_4 = tf.nn.dropout(pool_4, keep_prob=drop_conv, name='drop_4')

with tf.name_scope('Conv_5'):
    with tf.name_scope('Weights'):
        W_conv5 = tf.get_variable('W_conv5', shape=[3, 3, 64, 256])
        variable_summary(W_conv5)
    with tf.name_scope('Biases'):
        b_conv5 = tf.get_variable('b_conv5', shape=[256], initializer=tf.constant_initializer())
        variable_summary(b_conv5)
    with tf.name_scope('Wx_plus_b'):
        conv_5 = tf.nn.bias_add(tf.nn.conv2d(drop_4, W_conv5, strides=[1, 1, 1, 1], padding='VALID', name='conv_5'), b_conv5, name='conv_5_bias_add')
        tf.summary.histogram('Pre_activations', conv_5)
    a_5 = tf.nn.relu(conv_5, name='relu_conv_5')
    tf.summary.histogram('Activation', a_5)

with tf.name_scope('Conv_6'):
    with tf.name_scope('Weights'):
        W_conv6 = tf.get_variable('W_conv6', shape=[3, 3, 256, 256])
        variable_summary(W_conv6)
    with tf.name_scope('Biases'):
        b_conv6 = tf.get_variable('b_conv6', shape=[256], initializer=tf.constant_initializer())
        variable_summary(b_conv6)
    with tf.name_scope('Wx_plus_b'):
        conv_6 = tf.nn.bias_add(tf.nn.conv2d(a_5, W_conv6, strides=[1, 1, 1, 1], padding='VALID', name='conv_6'), b_conv6, name='conv_6_bias_add')
        tf.summary.histogram('Pre_activations', conv_6)
    a_6 = tf.nn.relu(conv_6, name='relu_conv_6')
    tf.summary.histogram('Activation', a_6)
    pool_6 = tf.nn.max_pool(a_6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_6')
    tf.summary.histogram('MaxPooling', pool_6)
    drop_6 = tf.nn.dropout(pool_6, keep_prob=drop_conv, name='drop_6')

with tf.name_scope('FC_7'):
    with tf.name_scope('Flattening'):
        drop_6_shape = drop_6.get_shape().as_list()
        flat_7 = tf.reshape(drop_6, shape=[-1, drop_6_shape[1]*drop_6_shape[2]*drop_6_shape[3]])
    with tf.name_scope('Weights'):
        W_fc7 = tf.get_variable('W_fc7', shape=[drop_6_shape[1]*drop_6_shape[2]*drop_6_shape[3], 1024])
        variable_summary(W_fc7)
    with tf.name_scope('Biases'):
        b_fc7 = tf.get_variable('b_fc7', shape=[1024], initializer=tf.constant_initializer())
        variable_summary(b_fc7)
    with tf.name_scope('Wx_plus_b'):
        h_fc7 = tf.nn.bias_add(tf.matmul(flat_7, W_fc7, name='fc_7_matmul'), b_fc7, name='fc_7_bias_add')
        tf.summary.histogram('Pre_activations', h_fc7)
    a_fc7 = tf.nn.relu(h_fc7, name='relu_fc_7')
    tf.summary.histogram('Activation', a_fc7)
    drop_7 = tf.nn.dropout(a_fc7, keep_prob=drop_fc, name='drop_7')

with tf.name_scope('FC_8'):
    with tf.name_scope('Weights'):
        W_fc8 = tf.get_variable('W_fc8', shape=[1024, 1024])
        variable_summary(W_fc8)
    with tf.name_scope('Biases'):
        b_fc8 = tf.get_variable('b_fc8', shape=[1024], initializer=tf.constant_initializer())
        variable_summary(b_fc8)
    with tf.name_scope('Wx_plus_b'):
        h_fc8 = tf.nn.bias_add(tf.matmul(drop_7, W_fc8, name='fc_8_matmul'), b_fc8, name='fc_8_bias_add')
        tf.summary.histogram('Pre_activations', h_fc8)
    a_fc8 = tf.nn.relu(h_fc8, name='relu_fc_8')
    tf.summary.histogram('Activation', a_fc8)
    drop_8 = tf.nn.dropout(a_fc8, keep_prob=drop_fc, name='drop_8')

with tf.name_scope('Out'):
    with tf.name_scope('Weights'):
        W_out = tf.get_variable('W_out', shape=[1024, NUM_CLASSES])
        variable_summary(W_out)
    with tf.name_scope('Biases'):
        b_out = tf.get_variable('b_out', shape=[NUM_CLASSES], initializer=tf.constant_initializer())
        variable_summary(b_out)
    with tf.name_scope('Wx_plus_b'):
        h_out = tf.nn.bias_add(tf.matmul(drop_8, W_out, name='out_matmul'), b_out, name='out_bias_add')
        tf.summary.histogram('Pre_activations', h_out)
    y_ = tf.nn.softmax(h_out, name='output')
    tf.summary.histogram('Final_Softmax', y_)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_Prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

with tf.name_scope('Loss'):
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_out, labels=y))
    tf.summary.scalar('Loss', mean_loss)

with tf.name_scope('Train'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('Learning_Rate'):
        learning_rate = tf.train.exponential_decay(learning_rate=LR, global_step=global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)
        tf.summary.scalar('Learning_Rate', learning_rate)
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM, use_nesterov=True).minimize(mean_loss, global_step=global_step)

aug = iaa.SomeOf(4, [
    iaa.Affine(rotate=(-30, 30)),
    iaa.Fliplr(1),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AddElementwise((-40, 40)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.5*255), per_channel=0.5)
])

saver = tf.train.Saver()
merged = tf.summary.merge_all()

max_acc = 0.
min_loss = 10.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_FOLDER+'train', sess.graph)
    val_writer = tf.summary.FileWriter(LOG_FOLDER+'val', sess.graph)
    tf.train.write_graph(sess.graph_def, '.', SAVE_FOLDER + 'tfsimpsons.pbtxt')
    X_val, y_val = dataset.val_set
    X_val = np.multiply(X_val, 1/255.)
    for epoch_i in range(EPOCHS):
        dataset.reset_batch_pointer()
        losses_val = []
        accuracies_val = []
        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
            X_batch, y_batch = dataset.next_batch
            X_batch = aug.augment_images(X_batch)
            X_batch = np.multiply(X_batch, 1/255.)
            summ_train, loss, _, acc_train = sess.run([merged, mean_loss, optimizer, accuracy], feed_dict={X: X_batch, y: y_batch, drop_conv: 0.80, drop_fc: 0.50})
            train_writer.add_summary(summ_train, tf.train.global_step(sess, global_step))
            train_writer.flush()
            if batch_num%25 == 0:
                print('{}/{}, Epoch: {}, Cost: {}, Accuracy: {}'.format(batch_num,
                                                                        EPOCHS * dataset.num_batches_in_epoch(),
                                                                        epoch_i, loss, acc_train))
        for batch_val_num, batch_val in enumerate(range(y_val.shape[0]//BATCH_SIZE)):
            acc_val, loss_val = sess.run([accuracy, mean_loss], feed_dict={X: X_val[BATCH_SIZE*batch_val_num:BATCH_SIZE*batch_val_num+1], y: y_val[BATCH_SIZE*batch_val_num:BATCH_SIZE*batch_val_num+1], drop_conv: 1.0, drop_fc: 1.0})
            accuracies_val.append(acc_val)
            losses_val.append(loss_val)
        losses_val = np.mean(losses_val)
        accuracies_val = np.mean(accuracies_val)
        summary = tf.Summary()
        summary.value.add(tag='Accuracy/Accuracy/Accuracy', simple_value=accuracies_val)
        summary.value.add(tag='Loss/Loss', simple_value=losses_val)
        val_writer.add_summary(summary, tf.train.global_step(sess, global_step))
        val_writer.flush()
        print('Epoch: {}, Validation Cost: {}, Validation Accuracy: {}\n'.format(epoch_i, losses_val, accuracies_val))
        if losses_val <= min_loss and accuracies_val >= max_acc:
            min_loss = losses_val
            max_acc = accuracies_val
            best_res = (epoch_i, min_loss, max_acc)
            print("Saving checkpoint...")
            saver.save(sess, SAVE_FOLDER + 'tfsimpsons.ckpt', global_step)
            print('Checkpoint saved in {}\n'.format(SAVE_FOLDER+'tfsimpsons.ckpt'))
        print('Best result\nEpoch: {}\nLoss: {}\nAccuracy: {}\n'.format(best_res[0], best_res[1], best_res[2]))
    X_test, y_test = dataset.test_set
    X_test = np.multiply(X_test, 1/255.)
    losses_test = []
    accuracies_test = []
    for batch_num, batch_i in enumerate(range(y_test.shape[0]//BATCH_SIZE)):
        acc_test, loss_test = sess.run([accuracy, mean_loss], feed_dict={X: X_test[BATCH_SIZE*batch_num:BATCH_SIZE*batch_num+1], y: y_test[BATCH_SIZE*batch_num:BATCH_SIZE*batch_num+1], drop_conv: 1.0, drop_fc: 1.0})
        losses_test.append(loss_test)
        accuracies_test.append(acc_test)
    losses_test = np.mean(losses_test)
    accuracies_test = np.mean(accuracies_test)
    print('Test Cost: {}, Test Accuracy: {}'.format(losses_test, accuracies_test))
