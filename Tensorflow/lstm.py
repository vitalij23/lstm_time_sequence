import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model, layers

matplotlib.use('TkAgg')

if __name__ == '__main__':
    BATCH_SIZE = 50
    STEPS = 20
    # LOAD
    # import torch
    # data = torch.load('traindata.pt')  # N=100, L=1000
    data = np.load('../123.npy', mmap_mode=None)[:, :8000].copy()
    print(data.shape)  # (2, 500) # price/value, steps

    # PREPARE
    test_st = 400  # test
    train_st = data.shape[1] - test_st  # train
    x_train = data[:, :train_st - 1]  # (2, 499)

    y_train = data[:, 1:train_st]  # (2, 499)

    x_train = np.transpose(x_train)
    x_train = np.expand_dims(x_train, 1)
    print("input", x_train.shape)
    y_train = np.transpose(y_train)
    y_train = np.expand_dims(y_train, 1)
    print("target", y_train.shape)

    # CUDA
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        # try:
        #     # Currently, memory growth needs to be the same across GPUs
        #     for gpu in gpus:
        #         tf.config.experimental.set_memory_growth(gpu, True)
        #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # except RuntimeError as e:
        #     # Memory growth must be set before GPUs have been initialized
        #     print(e)
        # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        # disable logger
        logging.getLogger('tensorflow').disabled = True

    # TRAIN
    print(x_train.shape)  # (1399, 1, 2)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(BATCH_SIZE, drop_remainder=True) #.prefetch(BATCH_SIZE)  # .shuffle(5000) repeat()
    # print(train_data.take(1))

    # Create LSTM Model.
    class LSTM(Model):
        # Set layers.
        def __init__(self):
            super(LSTM, self).__init__()
            # RNN (LSTM) hidden layer.
            self.lstm_layer1 = layers.LSTM(units=62, return_sequences=True, stateful=True)
            self.lstm_layer2 = layers.LSTM(units=262, return_sequences=True, stateful=True)
            self.lstm_layer3 = layers.LSTM(units=82, return_sequences=True, stateful=True)
            # self.lstm_layer4 = layers.LSTM(units=32, return_sequences=True, stateful=True)
            self.out = layers.Dense(2)

        # Set forward pass.
        def call(self, x, is_training=False):
            # LSTM layer.
            x = self.lstm_layer1(x)
            x = self.lstm_layer2(x)
            x = self.lstm_layer3(x)
            # x = self.lstm_layer4(x)

            # Output layer (num_classes).
            x = self.out(x)
            # if not is_training:
            #     # tf cross entropy expect logits without softmax, so only
            #     # apply softmax when not training.
            #     x = tf.nn.softmax(x)
            return x

    # Build LSTM model.
    lstm_net = LSTM()
    optimizer = tf.optimizers.Adamax(learning_rate=0.00002, beta_1=0.5) #SGD(learning_rate=0.006, decay=0.003, momentum=0.3)


    # Cross-Entropy Loss.
    # Note that this will apply 'softmax' to the logits.
    def my_loss(x, y):
        # Convert labels to int 64 for tf cross-entropy function.
        # y = tf.cast(y, tf.int64)
        # Apply softmax to logits and compute cross-entropy.
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        # print("x",y)
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)
        # loss3 = tf.nn.l2_loss(Y1 - Y2) * 2 / (reduce(lambda x, y: x * y, shape_obj))
        # tf.contrib.losses.softmax_cross_entropy
        loss = tf.keras.losses.MSLE(y, x)
        # print(loss)
        # Average loss across the batch.
        return tf.reduce_mean(loss)

    # Accuracy metric.
    def accuracy(y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    # Optimization process.
    def run_optimization(x, y):
        """

        :param x: [batch, timesteps, feature]
        :param y: [batch, timesteps, feature]
        :return:
        """
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = lstm_net(x, is_training=True)
            # Compute loss.
            loss = my_loss(pred, y)

        # Variables to update, i.e. trainable variables.
        trainable_variables = lstm_net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)

        # Update weights following gradients.
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss


    # Run training for the given number of steps.
    training_steps = 1000*100
    # display_step = 100
    for i in range(STEPS):
        for step, (batch_x, batch_y) in enumerate(train_data.take(-1), 1):  # (batch, steps, inputs)
            # Run the optimization to update W and b values.
            # print("step", step)
            # print(batch_x.shape)
            loss = run_optimization(batch_x, batch_y)

            # if step % display_step == 0:
            #     pred = lstm_net(batch_x, is_training=True)
                # acc = accuracy(pred, batch_y)
                # print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        print("%i, step: %i, loss: %f" % (i, step, loss))
        # lr = optimizer._decayed_lr(tf.float32)
        # print("lr: %f" % lr)
        lstm_net.lstm_layer1.reset_states()
        lstm_net.lstm_layer2.reset_states()
        lstm_net.lstm_layer3.reset_states()
        # lstm_net.lstm_layer4.reset_states()

    # lstm_net.lstm_layer.reset_states([np.zeros((1, 132)), np.zeros((1, 132))])
    print("TESTING")
    # 500, 2 -> 1, 1, 2
    data2 = np.expand_dims(data[:, 0], axis=0)
    data2 = np.expand_dims(data2, axis=0)
    print(data2.shape)

    # CHANGE BATCH SIZE BY CLONING
    old_weights = lstm_net.get_weights()
    lstm_net2 = LSTM()
    lstm_net2.build(data2.shape)
    # lstm_net.summary()
    lstm_net2.set_weights(old_weights)
    # 2, 500 -> 1, 500, 2
    data = np.transpose(data)
    data = np.expand_dims(data, axis=0)
    print(data.shape)
    lstm_net2.reset_states()

    res = lstm_net2(data[:, :data.shape[1]-test_st - 1, :], is_training=False)
    print("res", res.shape[1])

    p = np.expand_dims(data[:, test_st - 1, :], axis=0)
    p = lstm_net2.predict_on_batch(p)
    pred = [p]
    for i in range(test_st - 1):
        # n_x = np.expand_dims(data[:, train_st + i, :], axis=0)
        p = lstm_net2.predict_on_batch(p)
        pred.append(p)
    pred = np.array(pred)  # (401, 1, 1, 2)

    plt.plot(np.arange(0, data.shape[1]), data[0, :, 0], 'r', label="orig")
    plt.plot(np.arange(1, res.shape[1]+1), np.squeeze(res[:, :, 0]), 'b', label="train")
    l1 = res.shape[1]
    l2 = res.shape[1] + pred.shape[0]
    pr_len = np.arange(l1, l2)
    pr = np.squeeze(pred[:, :, :, 0])
    plt.plot(pr_len, pr, 'g', label="prediction")
    plt.show()
