import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model, layers

matplotlib.use('TkAgg')


class CNN_LSTM(Model):
    # Set layers. TODO: different gradient for cnn and lstm https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow?noredirect=1
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        # CNN
        self.c0 = layers.LocallyConnected2D(  # input (samples, rows, cols, channels)
            filters=4,
            kernel_size=2)
        # self.relu0 = layers.ReLU()
        self.mp0 = layers.AveragePooling2D(pool_size=(2, 1))
        self.d0 = layers.Dropout(0.1)
        self.c1 = layers.LocallyConnected2D(  # input (samples, rows, cols, channels)
            filters=8,
            kernel_size=2)
        # self.bn1 = layers.BatchNormalization(axis=1)
        self.relu1 = layers.ReLU()
        self.mp1 = layers.AveragePooling2D(pool_size=(2, 1))
        self.d1 = layers.Dropout(0.1)
        self.c2 = layers.LocallyConnected2D(  # input (samples, rows, cols, channels)
            filters=16,
            kernel_size=2)
        self.mp2 = layers.AveragePooling2D(pool_size=(2, 1))
        self.d2 = layers.Dropout(0.3)
        # self.bn2 = layers.BatchNormalization(axis=1)
        self.fl = layers.Flatten()
        self.relu2 = layers.ReLU()
        self.out_cnn = layers.Dense(25)

        # self.out_cnn = layers.Dense(1)

        # RNN (LSTM) hidden layer.
        self.lstm_layer1 = layers.LSTM(units=200, return_sequences=True, stateful=True, name="lstm1", dropout=0.3)
        self.lstm_layer2 = layers.LSTM(units=200, return_sequences=True, stateful=True, name="lstm2", dropout=0.3)
        # self.lstm_layer3 = layers.LSTM(units=150, return_sequences=True, stateful=True, name="lstm2", dropout=0.3)
        self.fl2 = layers.Flatten(name="lstm_fl")

        self.out_lstm = layers.Dense(25, name="lstm_d1")

        self.lstm_layer4 = layers.LSTM(units=200, return_sequences=True, stateful=True, dropout=0.3)
        self.fl3 = layers.Flatten()
        # self.final = layers.Dense(1024)
        self.out = layers.Dense(2)

    # Set forward pass.
    def call(self, inp, is_training=False):
        x, x2 = inp
        # x2 = inp
        # x = x2

        # CNN -------------------
        x = self.c0(x)
        # x = self.relu0(x)
        x = self.mp0(x)
        x = self.d0(x)
        x = self.c1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        x = self.d1(x)
        x = self.c2(x)
        x = self.mp2(x)
        x = self.d2(x)
        x = self.fl(x)
        x = self.relu2(x)
        x = self.out_cnn(x)

        if not is_training:  # tpremove CNN None batch dimension
            x = tf.reshape(x, [1, 25])

        # x = layers.Flatten(x)
        # Output layer (num_classes).
        # x = self.out(tf.nn.relu(x))

        # LSTM layer ---------------
        x2 = self.lstm_layer1(x2)
        x2 = self.lstm_layer2(x2)
        # x2 = self.lstm_layer3(x2)
        x2 = self.fl2(x2)
        x2 = self.out_lstm(x2)

        # concatenate -------------
        x = layers.concatenate([x2, x])

        x = tf.expand_dims(x, -2)
        x = self.lstm_layer4(x)
        x = self.fl3(x)

        # x = self.final(tf.nn.relu(x))
        x = self.out(tf.nn.relu(x))

        return x


def cuda():
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
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4800)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        # disable logger
        logging.getLogger('tensorflow').disabled = True


if __name__ == '__main__':
    BATCH_SIZE = 20
    STEPS = 2
    # LOAD
    # import torch
    # data = torch.load('traindata.pt')  # N=100, L=1000
    data = np.load('../123.npy', mmap_mode=None)
    print("data", data.shape)  # (2, 500) # price/value, steps

    # CNN
    l = np.load('../1sw.npz', mmap_mode=None)
    x_wind = l['x']  # steps, window_size, features
    y_wind = l['y']  # steps, features
    x_wind_least = l['x_without_y']  # steps, window_size, features
    w_size = l['w_size']
    future = l['future']
    file_path: str = str(l['file_path'])
    print(file_path)
    print(x_wind.shape, y_wind.shape, w_size, future)

    # PREPARE
    test_st = 0  # test
    train_st = x_wind.shape[0] - test_st  # train
    # CNN
    yw_train = y_wind[:, :, 0:3]  # batches, lstm/cnn, price/volume
    # print("wtf", yw_train.shape)
    xw_train: np.array = x_wind[:train_st]  # (2, 499)
    yw_train: np.array = yw_train[:train_st]  # (2, 499)
    xw_test: np.array = x_wind[train_st:]  # (2, 499)
    yw_test: np.array = y_wind[train_st:, :, 0:3]  # (2, 499)
    # xw_train = np.expand_dims(xw_train, -1)  # add channels
    # yw_train = np.expand_dims(yw_train, -1)
    print("xw_train", xw_train.shape)
    print("yw_train", yw_train.shape)

    # CUDA
    cuda()

    # TRAIN -----------
    # CNN
    # print("wtf", yw_train.shape)  # (1399, 1, 2)
    train_data_w = tf.data.Dataset.from_tensor_slices((xw_train, yw_train))
    train_data_w = train_data_w.batch(BATCH_SIZE, drop_remainder=True)  # .prefetch(BATCH_SIZE)  # .shuffle(5000) repeat()    # print(train_data.take(1))
    # a = train_data_w.take(1)
    # print("take", a.shape)
    # exit()

    # Build models.
    unio_net = CNN_LSTM()
    # optimizer = tf.optimizers.Adam(learning_rate=0.00008, beta_1=0.1, beta_2=0.1, epsilon=1e-3)
    # optimizer = tf.optimizers.SGD(learning_rate=0.015, decay=0.6, momentum=0.3)
    # optimizer_lstm = tf.optimizers.Adam(learning_rate=0.00001) #SGD(learning_rate=0.006, decay=0.003, momentum=0.3)
    # optimizer = tf.optimizers.Adam(learning_rate=0.00025)  # LSTM
    # optimizer = tf.optimizers.Adam(learning_rate=0.0001)  # CNN
    optimizer = tf.optimizers.Adam(learning_rate=0.00000951)

    # Accuracy metric.
    def accuracy(y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    # Optimization process.
    # pred_u = None

    def run_optimization(x, y):
        """

        :param x: [batch, window, features, channels]
        :param y: [batch, lstm_x/cnn, features, channels] (50, 2, 2, 1)
        :return:
        """
        global pred_u
        # Wrap computation inside a GradientTape for automatic differentiation.
        # print(y.shape)

        # Forward pass.
        x = np.expand_dims(x, -1)  # (50, 30, 5, 1)
        # print(x.shape)  # (50, 30, 5, 1)

        x_l = y[:, np.newaxis, 0, :]  # lstm
        y = y[:, 1, 0]  # price

        # y = y[:, 1, :]  # price and volume
        with tf.GradientTape() as tape:
            pred_u = unio_net((x, x_l), is_training=True)
            # pred_u = unio_net((x, pred_u), is_training=True)
            # print("pred_u", pred_u.shape)
            # pred_u = unio_net(x_l, is_training=True)
            # c = layers.add([pred_lstm, pred_cnn])
            # print(c.shape)

            # Compute loss.
            loss = tf.keras.losses.MSLE(y, pred_u[:, 0])
            loss = tf.reduce_mean(loss)
            # loss = my_loss(pred_u[:, 0], y)

        # Variables to update, i.e. trainable variables.
        # lstm_trainable_variables = []
        # other_trainable_variables = []
        # for l in unio_net.layers:
        #     if "lstm" in l.name:
        #         # print(l.name)
        #         lstm_trainable_variables += l.trainable_variables
        #     else:
        #         other_trainable_variables += l.trainable_variables



        # Compute gradients.
        # gradients = g.gradient(loss, lstm_trainable_variables + other_trainable_variables)
        # grads1 = gradients[:len(lstm_trainable_variables)]
        # grads2 = gradients[len(lstm_trainable_variables):]
        #
        # # Update weights following gradients.
        # optimizer_lstm.apply_gradients(zip(grads1, lstm_trainable_variables))
        # optimizer_othr.apply_gradients(zip(grads2, other_trainable_variables))
        # ONE
        trainable_variables_union = unio_net.trainable_variables
        gradients = tape.gradient(loss, trainable_variables_union)
        optimizer.apply_gradients(zip(gradients, trainable_variables_union))
        return loss

    # TRAIN
    for i in range(STEPS):
        # pred_u = np.zeros((BATCH_SIZE, 2))
        for step, (batch_x, batch_y) in enumerate(train_data_w.take(-1)):  # (batch, steps, inputs)
            # print("s", step * 20 + w_size + future)
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
        unio_net.lstm_layer1.reset_states()
        unio_net.lstm_layer2.reset_states()
        # unio_net.lstm_layer3.reset_states()
        unio_net.lstm_layer4.reset_states()

    print("TESTING")
    # xw_train (400, 30, 5)
    # yw_train (400, 2, 2)
    # (50, 30, 5, 1)
    # (50, 2, 2)
    # (1, 3, 5, 1)
    # (1, 1, 1)
    tx_0 = np.expand_dims(xw_train[0, :, :, np.newaxis], axis=0)
    ty_0 = yw_train[0][0]
    ty_0 = ty_0[np.newaxis, np.newaxis]
    # print(tx_0.shape, ty_0.shape)

    # CHANGE BATCH SIZE BY CLONING
    old_weights = unio_net.get_weights()
    keras.backend.clear_session()

    unio_net2 = CNN_LSTM()
    unio_net2.build(input_shape=[tx_0.shape, ty_0.shape])  # (1, 30, 5) (1, 1, 2)
    # unio_net2.summary()
    unio_net2.set_weights(old_weights)
    unio_net2.reset_states()

    # FULL PREDICT
    for i in range(2, 183, 60):
        # TRAIN OUT unroll to be prepared for predict
        res = []
        # r = np.zeros((1, 2))
        na = xw_train.shape[0] - xw_train.shape[0]//i
        for i in range(na, xw_train.shape[0]):
            x = xw_train[i]
            y = yw_train[i]  # 2, 2
            x1 = x[np.newaxis, :, :, np.newaxis]
            # print(y.shape)
            x2 = y[np.newaxis, np.newaxis, 0, :]
            # r = r[:, np.newaxis, :]
            # print(x1.shape, x2.shape)  # (30, 5) (2, 2)
            r = unio_net2.predict_on_batch([x1, x2])
            res.append(r)
        res = np.array(res)
        print(res.shape)  # (400, 1, 2)
        train = res[:, 0, 0]
        # print(train.shape)

        # TEST ( optional)
        if test_st != 0:
            res = []
            for i in range(xw_test.shape[0]):
                x = xw_test[i]
                y = yw_test[i]  # 2, 2
                x1 = x[np.newaxis, :, :, np.newaxis]
                # print(y.shape)
                x2 = y[np.newaxis, np.newaxis, 0, :]
                # r = r[:, np.newaxis, :]
                r = unio_net2.predict_on_batch([x1, x2])
                # r = unio_net2.predict_on_batch([x1, r])
                res.append(r)
            res = np.array(res)
            # print(res.shape)  # (400, 1, 2)
            test = res[:, 0, 0]
            # print(test.shape)

        # PREDICT
        res2 = []
        # print(r.shape)
        for i in range(x_wind_least.shape[0]):
            x = x_wind_least[i]
            x1 = x[np.newaxis, :, :, np.newaxis]
            r = r[:, np.newaxis, :]
            # print(r.shape)
            # print(x1.shape, x2.shape)  # (30, 5) (2, 2)
            r = unio_net2.predict_on_batch([x1, r])
            # print(r.shape)
            res2.append(r)
        res2 = np.array(res2)
        # print(res.shape)  # (400, 1, 2)
        pre = res2[:, 0, 0]
        # print(pre.shape)
        plt.plot(np.arange(train_st + w_size + future + xw_test.shape[0], train_st + w_size + future + xw_test.shape[0] + len(pre)), pre, 'k', label="predict")

    plt.plot(np.arange(0, data.shape[1]), data[0, :], 'r', label="orig")
    # plt.plot(np.arange(w_size + future, train_st + w_size + future), train, 'g', label="test")
    if xw_test.shape[0] != 0:
        plt.plot(np.arange(train_st + w_size + future, train_st + w_size + future + xw_test.shape[0]), test, 'b', label="test free")

    # plt.plot(np.arange(train_st + w_size + future + xw_test.shape[0], train_st + w_size + future + xw_test.shape[0] + len(pre)), pre, 'k', label="predict")
    plt.legend()
    import os
    plt.savefig(os.path.basename(file_path)+'.jpg')
    plt.show()

    # res = lstm_net2(data[:, :data.shape[1]-test_st - 1, :], is_training=False)
    # print("res", res.shape[1])
    #
    # p = np.expand_dims(data[:, test_st - 1, :], axis=0)
    # p = lstm_net2.predict_on_batch(p)
    # pred = [p]
    # for i in range(test_st - 1):
    #     # n_x = np.expand_dims(data[:, train_st + i, :], axis=0)
    #     p = lstm_net2.predict_on_batch(p)
    #     pred.append(p)
    # pred = np.array(pred)  # (401, 1, 1, 2)

    # plt.plot(np.arange(0, data.shape[1]), data[0, :, 0], 'r', label="orig")
    # plt.plot(np.arange(1, res.shape[1]+1), np.squeeze(res[:, :, 0]), 'b', label="train")
    # l1 = res.shape[1]
    # l2 = res.shape[1] + pred.shape[0]
    # pr_len = np.arange(l1, l2)
    # pr = np.squeeze(pred[:, :, :, 0])
    # plt.plot(pr_len, pr, 'g', label="prediction")
    # plt.show()
