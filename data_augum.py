import csv
import numpy as np
import torch


def strict_windows(arr: list, batch_num=200) -> list:
    batch_num = 200
    batches = []
    for di in range(0, len(arr), batch_num):
        window = arr[di:di + batch_num]
        if len(window) < 1000:  # last short window
            window = list(window) + [0.] * (batch_num - len(window))
        batches.append(window)
    return batches


def striding_windows(arr: list, batch_num=200) -> list:
    batches = []
    for di in range(len(arr) - batch_num + 1):
        window = arr[di:di + batch_num]
        batches.append(window)
    return batches


def striding_windows_reverse(arr) -> np.array:
    """
    :param arr: first dimensions - batch, second - windows
    :return: flat array
    """
    batches = []
    for window in arr:
        if not batches:  # first window
            batches.append(window)  # whole
        else:
            batches.append(window[-1:])  # last one
    return np.concatenate(batches)


def scaler(data: np.array, axis: int) -> np.array:
    """ in range (0,1)

    :param data: two dimensions
    :return:(0,1)
    """
    data_min = np.nanmin(data[:, axis])
    data_max = np.nanmax(data[:, axis])
    data[:, axis] = (data[:, axis] - data_min) / (data_max - data_min)
    return data


def scaler_simple(data: np.array) -> np.array:
    """ in range (0,1)

    :param data: one dimensions
    :return:(0,1)
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data = (data - data_min) / (data_max - data_min)
    return data


def my_scaler(data: np.array) -> np.array:
    """ data close to 0 will not add much value to the learning process

    :param data: two dimensions 0 - time, 1 - prices
    :return:
    """

    # data = scaler(data, axis=0)
    smoothing_window_size = data.shape[0]//2  # for 10000 - 4
    dl = []
    for di in range(0, len(data), smoothing_window_size):
        window = data[di:di + smoothing_window_size]
        # print(window.shape)
        window = scaler(window, axis=1)
        # print(window[0], window[-1])
        dl.append(window)  # last line will be shorter

    return np.concatenate(dl)


def augum(path: str, rows_numers: list, scaling=True) -> np.array:
    """
    :param path:
    :param rows_numers:
    :param scaling:
    :return: [[row1, row2]...]
    """

    data = []

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # line = []
            for x in rows_numers:
                time = float(int(row[2]) + float(row[3])/(10**6))
                line = (time, float(row[x]))
                data.append(line)  # list of selected rows
    return np.array(data)


if __name__ == '__main__':
    # 9 rows
    p = '/mnt/hit4/hit4user/PycharmProjects/my_pytorch_lstm/YNDX_191211_191223.csv'
    data = augum(p, [7], scaling=True)
    data = data[:2000, :]  # limit
    # replace date
    dat = list(range(data.shape[0]))
    data[:, 0] = scaler_simple(dat)

    # SCALING
    data = my_scaler(data)  # (0,1)
    # save original
    torch.save(data, open('traindata_ya2.pt', 'wb'))

    # BATCHING

    # 1) strict windows
    # for di in range(0, len(data), batch_num):
    #     window = train_data[di:di + batch_num]
    #     if len(window) < 1000:  # last short window
    #         window = list(window) + [0.] * (batch_num - len(window))
    #     batches.append(window)
    # 2) slide window with stride 1
    # batches = striding_windows(list(data), batch_num=100)  # two dimension

    # 3) one step timesteps batching
    # from batching_step import DataGeneratorSeq
    #
    # dg = DataGeneratorSeq(data, batch_size=50, num_unroll=300)
    # u_data, u_labels = dg.unroll_batches()
    # u_data, u_labels = np.array(u_data), np.array(u_labels)
    # 4)
    from batching_gap import batches
    u_data, u_labels = batches(data, 150, 900, offset=100, second_offset=2)

    print(u_data.shape)  # (300, 10, 2) # steps, batchs, time/price
    print(u_labels.shape)
    batches = np.stack([u_data, u_labels], axis=1)
    batches = [u_data, u_labels]
    # print(batches)

    data_res = np.array(batches)  # (2, 300, 10, 2) # train/test, steps, batchs, time/price
    print(data_res.shape)
    torch.save(data_res, open('traindata_ya.pt', 'wb'))


    # PRINT
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')

    # plt.plot(np.arange(data.shape[0]), data[:, 1], 'b', linewidth=2.0)
    # plt.show()

    # from matplotlib import pyplot as plt
    # plt.plot(range(len(train_data)), a)
    # plt.show()


    # a1 = scaler(d[1])
    # print(a1)
    # print(d)
# data = scale10(data)
# print(np.mean(data), max(data), min(data))

#
# # print(data_min, data_max)
# scale = np.nanstd(data, axis=0)
# data /= scale
# print(np.nanstd(data, axis=0))
# mean = np.nanmean(data, axis=0)
# data -= mean
# print(np.nanmean(data, axis=0))
# print(np.nanstd(data, axis=0))
# #
# # print(data)