# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
# load data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# create a graph and then launch it in a session

sess = tf.InteractiveSession()
# We start building the computation graph
# by creating nodes for the input images and target output classes.
x = tf.placeholder(tf.float32, shape=[None, 28*28], name='input_images')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='target_output_classes')


# Weight Initialization
def weight_variable(shape):
    # tra ve 1 gia tri random theo thuat toan truncated_ normal
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_varibale(shape):
    initial = tf.constant(0.1, shape=shape, name='Bias')
    return tf.Variable(initial)


# Convolution and Pooling
def conv2d(x, W):
    # Must have `strides[0] = strides[3] = 1 `.
    # For the most common case of the same horizontal and vertices strides, `strides = [1, stride, stride, 1] `.
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME', name='conv_2d')


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='max_pool')


# Firt convolutional layer

# The first two dimensions are the patch size
# the next is the number of input channels, (chi co 1 khung mau di vo)
# and the last is the number of output channels ( R,G,B,...)->filters bank
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_varibale([32])
# To apply the layer, we first reshape x to a 4d tensor
# with the second and third dimensions corresponding to image width and height (28x28)
# and the final dimension corresponding to the number of color channels(1 vi luc dau vo)
# tham so dau tien la (-1) de doi cac chieu con lai vao
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

# We then convolve x_image with the weight tensor,
# add the bias,
# apply the ReLU function,
# and finally max pool. -> hidden layer

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
# In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Now that the image size has been reduced to 7x7
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
# We reshape the tensor from the pooling layer into a batch of vectors,
#  multiply by a weight matrix,
# add a bias,
# and apply a ReLU.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_varibale([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# To reduce overfitting, we will apply dropout before the readout layer.
#  We create a placeholder for the probability that a neuron's output is kept during dropout.
#  This allows us to turn dropout on during training,
#  and turn it off during testing.
# TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs
# in addition to masking them, so dropout just works without any additional scaling.1

# con nen giu lai cai element nay khong
# (keep_prob: A scalar `Tensor` with the same type as x : tensor. The probability
#       that each element is kept.)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# Readout Layer
# Finally, we add a layer, just like for the one layer softmax regression above.

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_varibale([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
# We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimize
# We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
# We will add logging to every 100th iteration in the training process.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={
        x: batch[0],
        y_: batch[1],
        keep_prob: 0.5
    })
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 1.0
        })
        print("step %d, training accuracy %g" % (i, train_accuracy))
        print(type(y_))


final_accuracy = accuracy.eval(feed_dict={
    x: mnist.test.images,
    y_: mnist.test.labels,
    keep_prob: 1.0
})

print("test accuracy %g" % final_accuracy)
