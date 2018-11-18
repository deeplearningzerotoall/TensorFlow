import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import mnist
from time import time

def normalize(X_train, X_test):
    """
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    """
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

def network(x, reuse=False) :
    xavier = tf_contrib.layers.xavier_initializer()

    with tf.variable_scope('network', reuse=reuse) :
        x = tf.layers.flatten(x)

        for i in range(4) :
            x = tf.layers.dense(inputs=x, units=512, use_bias=True, kernel_initializer=xavier, name='fully_connected_' + str(i))
            x = tf.nn.relu(x)

        hypothesis = tf.layers.dense(inputs=x, units=10, use_bias=True, kernel_initializer=xavier, name='fully_connected_logit')

        return hypothesis


""" dataset """
train_x, train_y, test_x, test_y = load_mnist()

""" parameters """
learning_rate = 0.001
batch_size = 128

training_epochs = 1
training_iterations = len(train_x) // batch_size

img_size = 28
c_dim = 1
label_dim = 10

""" Graph Input """
train_inptus = tf.placeholder(tf.float32, [batch_size, img_size, img_size, c_dim], name='train_inputs')
train_labels = tf.placeholder(tf.float32, [batch_size, label_dim], name='train_labels')

test_inptus = tf.placeholder(tf.float32, [None, img_size, img_size, c_dim], name='test_inputs')
test_labels = tf.placeholder(tf.float32, [None, label_dim], name='test_labels')

""" Model """
train_logits = network(train_inptus)
test_logits = network(test_inptus, reuse=True)

train_loss, train_accuracy = classification_loss(logit=train_logits, label=train_labels)
_, test_accuracy = classification_loss(logit=test_logits, label=test_labels)

""" Training """
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)


with tf.Session() as sess :
    tf.global_variables_initializer().run()
    start_time = time()

    """ Training phase """
    for epoch in range(training_epochs) :
        for idx in range(training_iterations) :
            batch_x = train_x[idx * batch_size:(idx + 1) * batch_size]
            batch_y = train_y[idx * batch_size:(idx + 1) * batch_size]

            train_feed_dict = {
                train_inptus: batch_x,
                train_labels: batch_y
            }

            # update network
            _, train_loss_val, train_accuracy_val = sess.run([optimizer, train_loss, train_accuracy], feed_dict=train_feed_dict)
            print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.8f, train_accuracy: %.2f" \
                  % (epoch, idx, training_iterations, time() - start_time, train_loss_val, train_accuracy_val))

    print('Learning Finished!')

    """ Test phase """
    test_feed_dict = {
        test_inptus: test_x,
        test_labels: test_y
    }

    test_accuracy_val = sess.run(test_accuracy, feed_dict=test_feed_dict)
    print("Test accuracy: %.8f" % (test_accuracy_val) )

    """ Get test image """
    r = np.random.randint(low=0, high=len(test_x) - 1)
    print("Label: ", np.argmax(test_y[r: r+1], axis=-1))
    print("Prediction: ", sess.run(tf.argmax(test_logits, axis=-1), feed_dict={test_inptus: test_x[r: r+1]}))

    plt.imshow(test_x[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
