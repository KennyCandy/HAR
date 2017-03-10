# Thanks to Zhao Yu for converting the .ipynb notebook to
# this simplified Python script that I edited a little.

# Note that the dataset must be already downloaded for this script to work, do:
#     $ cd data/
#     $ python download_dataset.py

import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

import os

if __name__ == "__main__":

    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    # Preparing data set:
    TRAIN = "train/"
    TEST = "test/"


    # Load "X" (the neural network's training and testing inputs)
    def load_X(X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'rb')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                    ]]
            )
            file.close()

        """Examples
        --------
        >> > x = np.arange(4).reshape((2, 2))
        >> > x
        array([[0, 1],
               [2, 3]])

        >> > np.transpose(x)
        array([[0, 2],
               [1, 3]])

        >> > x = np.ones((1, 2, 3))
        >> > np.transpose(x, (1, 0, 2)).shape
        (2, 1, 3)
        """

        return np.transpose(np.array(X_signals), (1, 2, 0))


    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    # print(X_train)
    print(len(X_train))  # 7352
    print(len(X_train[0]))  # 128
    print(len(X_train[0][0]))  # 9

    print(type(X_train))

    X_train = np.reshape(X_train, [-1, 64, 18])
    X_test = np.reshape(X_test, [-1, 64, 18])

    # print(X_train)
    print(len(X_train))  # 7352
    print(len(X_train[0]))  # 128
    print(len(X_train[0][0]))  # 9

    print(type(X_train))
    # exit()

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"


    def one_hot(label):
        """convert label from dense to one hot
          argument:
            label: ndarray dense label ,shape: [sample_num,1]
          return:
            one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
        """
        label_num = len(label)
        new_label = label.reshape(label_num)  # shape : [sample_num]
        # because max is 5, and we will create 6 columns
        n_values = np.max(new_label) + 1
        return np.eye(n_values)[np.array(new_label, dtype=np.int32)]


    # Load "y" (the neural network's training and testing outputs)
    def load_y(y_path):
        file = open(y_path, 'rb')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
                ]],
            dtype=np.int32
        )
        file.close()
        # Subtract 1 to each output class for friendly 0-based indexing
        return y_ - 1


    y_train = one_hot(load_y(y_train_path))
    y_test = one_hot(load_y(y_test_path))

    print("---------y_train----------")
    # print(y_train)
    print(len(y_train))  # 7352
    print(len(y_train[0]))  # 6
    
                           
    # exit()

    # -----------------------------------
    # step2: define parameters for model
    # -----------------------------------
    class Config(object):
        """
        define a class to store parameters,
        the input should be feature mat of training and testing
        """

        def __init__(self, X_train, X_test):
            # Input data
            self.train_count = len(X_train)  # 7352 training series
            self.test_data_count = len(X_test)  # 2947 testing series
            self.n_steps = len(X_train[0])  # 128 time_steps per series

            # Trainging
            self.learning_rate = 0.0025
            self.lambda_loss_amount = 0.0015
            self.training_epochs = 300
            self.batch_size = 700

            # LSTM structure
            self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
            self.n_hidden = 32  # nb of neurons inside the neural network
            self.n_classes = 6  # Final output classes
            self.W = {
                'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
                'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
            }
            self.biases = {
                'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
                'output': tf.Variable(tf.random_normal([self.n_classes]))
            }


    config = Config(X_train, X_test)
    # print("Some useful info to get an insight on dataset's shape and normalisation:")
    # print("features shape, labels shape, each features mean, each features standard deviation")
    # print(X_test.shape, y_test.shape,
    #       np.mean(X_test), np.std(X_test))
    # print("the dataset is therefore properly normalised, as expected.")
    #
    #
    # ------------------------------------------------------
    # step3: Let's get serious and build the neural network
    # ------------------------------------------------------
    # [none, 128, 9]
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    # [none, 6]
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    print("-------X Y----------")
    print(X)
    X = tf.reshape(X, shape=[-1, 64, 18])
    print(X)

    print(Y)
    Y = tf.reshape(Y, shape=[-1, 6])
    print(Y)

    # [START]ADD CNN
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
    x_image = tf.reshape(x, shape=[-1, 64, 18, 1])


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
    # h_pool2 = max_pool_2x2(h_conv2)
    h_pool2 = h_conv2

    # Now that the image size has been reduced to 7x7
    # we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
    # We reshape the tensor from the pooling layer into a batch of vectors,
    #  multiply by a weight matrix,
    # add a bias,
    # and apply a ReLU.

    W_fc1 = weight_variable([32 * 9 * 64, 1024])
    b_fc1 = bias_varibale([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 9 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    ###
    # Readout Layer
    # Finally, we add a layer, just like for the one layer softmax regression above.

    W_fc2 = weight_variable([1024, 6])
    b_fc2 = bias_varibale([6])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    # [END]ADD CNN 

    def LSTM_Network(feature_mat, config):
        """model a LSTM Network,
          it stacks 2 LSTM layers, each layer has n_hidden=32 cells
           and 1 output layer, it is a full connet layer
          argument:
            feature_mat: ndarray feature matrix, shape=[batch_size,time_steps,n_inputs]
            config: class containing config of network
          return:
                  : matrix  output shape [batch_size,n_classes]
        """
        # Exchange dim 1 and dim 0
        feature_mat = tf.transpose(feature_mat, [1, 0, 2])
        # New feature_mat's shape: [time_steps, batch_size, n_inputs]

        # Temporarily crush the feature_mat's dimensions
        feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])  # 9
        # New feature_mat's shape: [time_steps*batch_size, n_inputs]  # 128 * batch_size

        # Linear activation, reshaping inputs to the LSTM's number of hidden:
        hidden = tf.nn.relu(tf.matmul(
            feature_mat, config.W['hidden']
        ) + config.biases['hidden'])
        # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden]

        # Split the series because the rnn cell needs time_steps features, each of shape:
        hidden = tf.split(0, config.n_steps, hidden)
        # New hidden's shape: a list of length "time_step" containing tensors of shape [batch_size, n_hidden]

        # Define LSTM cell of first hidden layer:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)

        # Stack two LSTM layers, both layers has the same shape
        lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)

        # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
        outputs, _ = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)
        # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_classes]

        # Get last time step's output feature for a "many to one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


    pred_Y = LSTM_Network(X, config)
    print(pred_Y)

    # Loss,train_step,evaluation
    l2 = config.lambda_loss_amount * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    train_step = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    # --------------------------------------------
    # step4: Hooray, now train the neural network
    # --------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.initialize_all_variables().run()

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),  # (0, 7352, 1500)
                              range(config.batch_size, config.train_count + 1,
                                    config.batch_size)):  # (1500, 7353, 1500)
            print(start)
            print(end)

            sess.run(train_step, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
            X: X_test, Y: y_test})
        print("traing iter: {},".format(i) + \
              " test accuracy : {},".format(accuracy_out) + \
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
    #
    # #------------------------------------------------------------------
    # # step5: Training is good, but having visual insight is even better
    # #------------------------------------------------------------------
    # # The code is in the .ipynb
    #
    # #------------------------------------------------------------------
    # # step6: And finally, the multi-class confusion matrix and metrics!
    # #------------------------------------------------------------------
    # # The code is in the .ipynb
