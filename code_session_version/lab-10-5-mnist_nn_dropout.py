import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from time import time
import os

def save(sess, saver, checkpoint_dir, model_name, step):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)


def load(sess, saver, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt :
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def normalize(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data = np.expand_dims(test_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10) # [N,] -> [N, 10]
    test_labels = to_categorical(test_labels, 10) # [N,] -> [N, 10]

    return train_data, train_labels, test_data, test_labels

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

def network(x, dropout_rate=0, reuse=False) :
    xavier = tf_contrib.layers.xavier_initializer()

    with tf.variable_scope('network', reuse=reuse) :
        x = tf.layers.flatten(x) # [N, 28, 28, 1] -> [N, 784]

        for i in range(4) :
            # [N, 784] -> [N, 512] -> [N, 512] -> [N, 512] -> [N, 512]
            x = tf.layers.dense(inputs=x, units=512, use_bias=True, kernel_initializer=xavier, name='fully_connected_' + str(i))
            x = tf.nn.relu(x)
            x = tf.layers.dropout(inputs=x, rate=dropout_rate)

        # [N, 512] -> [N, 10]
        hypothesis = tf.layers.dense(inputs=x, units=10, use_bias=True, kernel_initializer=xavier, name='fully_connected_logit')

        return hypothesis # hypothesis = logit


""" dataset """
train_x, train_y, test_x, test_y = load_mnist()

""" parameters """
learning_rate = 0.001
batch_size = 128

training_epochs = 10
training_iterations = len(train_x) // batch_size

img_size = 28
c_dim = 1
label_dim = 10

train_flag = True

""" Graph Input using Dataset API """
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=batch_size).\
    batch(batch_size).\
    repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x)).\
    repeat()

""" Dropout rate"""
dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

""" Model """
train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

train_inputs, train_labels = train_iterator.get_next()
test_inputs, test_labels = test_iterator.get_next()

train_logits = network(train_inputs)
test_logits = network(test_inputs, reuse=True)

train_loss, train_accuracy = classification_loss(logit=train_logits, label=train_labels)
_, test_accuracy = classification_loss(logit=test_logits, label=test_labels)

""" Training """
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)

"""" Summary """
summary_train_loss = tf.summary.scalar("train_loss", train_loss)
summary_train_accuracy = tf.summary.scalar("train_accuracy", train_accuracy)

summary_test_accuracy = tf.summary.scalar("test_accuracy", test_accuracy)

train_summary = tf.summary.merge([summary_train_loss, summary_train_accuracy])
test_summary = tf.summary.merge([summary_test_accuracy])


with tf.Session() as sess :
    tf.global_variables_initializer().run()
    start_time = time()

    saver = tf.train.Saver()
    checkpoint_dir = 'checkpoints'
    logs_dir = 'logs'

    model_dir = 'nn_dropout'
    model_name = 'dense'

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    logs_dir = os.path.join(logs_dir, model_dir)

    if train_flag :
        writer = tf.summary.FileWriter(logs_dir, sess.graph)
    else :
        writer = None

    # restore check-point if it exits
    could_load, checkpoint_counter = load(sess, saver, checkpoint_dir)

    if could_load:
        start_epoch = (int)(checkpoint_counter / training_iterations)
        start_batch_index = checkpoint_counter - start_epoch * training_iterations
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        start_epoch = 0
        start_batch_index = 0
        counter = 1
        print(" [!] Load failed...")

    if train_flag :
        """ Training phase """
        for epoch in range(start_epoch, training_epochs) :
            for idx in range(start_batch_index, training_iterations) :

                train_feed_dict = {
                    dropout_rate: 0.3
                }

                test_feed_dict = {
                    dropout_rate: 0.0
                }

                # train
                _, summary_str, train_loss_val, train_accuracy_val = sess.run([optimizer, train_summary, train_loss, train_accuracy], feed_dict=train_feed_dict)
                writer.add_summary(summary_str, counter)

                # test
                summary_str, test_accuracy_val = sess.run([test_summary, test_accuracy], feed_dict=test_feed_dict)
                writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.8f, train_accuracy: %.2f, test_Accuracy: %.2f" \
                      % (epoch, idx, training_iterations, time() - start_time, train_loss_val, train_accuracy_val, test_accuracy_val))

            start_batch_index = 0
            save(sess, saver, checkpoint_dir, model_name, counter)

        save(sess, saver, checkpoint_dir, model_name, counter)
        print('Learning Finished!')

        test_feed_dict = {
            dropout_rate: 0.0
        }

        test_accuracy_val = sess.run(test_accuracy, feed_dict=test_feed_dict)
        print("Test accuracy: %.8f" % (test_accuracy_val))

    else :
        """ Test phase """
        test_feed_dict = {
            dropout_rate: 0.0
        }

        test_accuracy_val = sess.run(test_accuracy, feed_dict=test_feed_dict)
        print("Test accuracy: %.8f" % (test_accuracy_val))

        """ Get test image """
        r = np.random.randint(low=0, high=len(test_x) - 1)
        print("Label: ", np.argmax(test_y[r: r+1], axis=-1))
        print("Prediction: ", sess.run(tf.argmax(test_logits, axis=-1), feed_dict={test_inputs: test_x[r: r+1]}))

        plt.imshow(test_x[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()
