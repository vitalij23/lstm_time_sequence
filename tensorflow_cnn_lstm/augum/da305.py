import csv
import numpy as np
# import torch
import matplotlib.pyplot as plt
import matplotlib
# own
from augum.savitzky_golay import savitzky_golay
from augum.batching_cnn_lstm import striding_windows_with_future

matplotlib.use('TkAgg')


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


def strict_windows(arr: list, batch_num=200) -> list:
    batch_num = 200
    batches = []
    for di in range(0, len(arr), batch_num):
        window = arr[di:di + batch_num]
        if len(window) < 1000:  # last short window
            window = list(window) + [0.] * (batch_num - len(window))
        batches.append(window)
    return batches


def striding_windows(arr: list, batch_num=200) -> np.array:
    """window slide by 1 form 0 to len-w_size """
    batches = []
    for di in range(len(arr) - batch_num + 1):
        window = arr[di:di + batch_num]
        print(di, window)
        batches.append(window)
    return np.array(batches)


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
    smoothing_window_size = data.shape[0]  # // 2  # for 10000 - 4
    dl = []
    for di in range(0, len(data), smoothing_window_size):
        window = data[di:di + smoothing_window_size]
        # print(window.shape)
        window = scaler(window, axis=1)
        # print(window[0], window[-1])
        dl.append(window)  # last line will be shorter

    return np.concatenate(dl)


def augum(path: str, rows_numers: list) -> np.array:
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

            line = []
            # time = 0 #float(int(row[2]) + float(row[3]) / (10 ** 6))
            # line.append(time)
            for x in rows_numers:
                line.append(float(row[x]))
            data.append(line)  # list of selected rows
    return np.array(data)


def main():
    # 9 rows
    p = '/home/u2/Downloads/ALRS_170123_200130.csv'
    lim = 200
    data_full = augum(p, [7, 8, 4, 5, 6])  # steps, time/price/volume
    print(data_full.shape)  # (6819, 3)
    # replace date
    dat = list(range(data_full.shape[0]))
    # data_full[:, 0] = scaler_simple(dat)
    # TEST replace price
    # data_full[:, 0] = scaler_simple(dat)

    # SELECT DATA FOR LSTM - price/volume
    data = data_full[:, 0:2].copy()

    # SCALING
    # data = my_scaler(data)  # (0,1)
    data[:, 0] = scaler_simple(data[:, 0])  # price
    data[:, 1] = scaler_simple(data[:, 1])  # value
    # CNN scalling
    for d in range(data_full.shape[1]):
        data_full[:, d] = scaler_simple(data_full[:, d])  # value
        data_full[:, d] = savitzky_golay(data_full[:, d], 11, 3)

    # SMOOTHING
    data[:, 0] = savitzky_golay(data[:, 0], 11, 3)
    data[:, 1] = savitzky_golay(data[:, 1], 11, 3)
    # PLOT
    # plt.plot(np.arange(data.shape[0]), s, 'r', linewidth=2.0)
    # plt.plot(np.arange(data.shape[0]), data[:, 1], 'b', linewidth=2.0)
    # plt.show()
    # SAVE FOR LSTM
    data_t = data.transpose((1, 0))  # (2, 6819)
    # print(data)
    print(data_t.shape)  # 2, 500 #price/volume, step
    np.save('../123', data_t)
    # torch.save(data, open('../traindata.pt', 'wb'))

    # PRINT origin

    # plt.plot(np.arange(data.shape[0]), s, 'r', linewidth=2.0)
    # plt.plot(np.arange(data.shape[0]), data[:, 1], 'b', linewidth=2.0)
    # plt.show()

    # image

    # BATCHING

    # 1) strict windows
    # for di in range(0, len(data), batch_num):
    #     window = train_data[di:di + batch_num]
    #     if len(window) < 1000:  # last short window
    #         window = list(window) + [0.] * (batch_num - len(window))
    #     batches.append(window)
    # 2) slide window with stride 1 with future
    # print(data.shape)

    limit = data_full.shape[0] - lim  # for testing
    future = 250
    w_size = 30
    ret = striding_windows_with_future(data_full[:limit], w_size=w_size, future=future)  # two dimension
    print(ret['x'].shape, ret['y'].shape, ret['x_without_y'].shape)
    np.savez('../1sw', x=ret['x'], y=ret['y'], x_without_y=ret['x_without_y'], w_size=w_size, future=future)
    # print(ret['y'][:, 0, 0])
    plt.plot(range(data_full.shape[0]), data_full[:, 0], 'b')
    plt.plot(range(w_size+future, len(ret['y'][:, 0, 0]) + w_size+future), ret['y'][:, 0, 0], 'r')
    plt.show()
    return ret['x'], ret['y'], ret['x_without_y'], w_size, future


if __name__ == '__main__':
    batches = main()
    # print("b1", batches[0].shape)  # 23 - windows, 2 - window/future
    # print("b", batches[0][0][0].shape)
    # np.savez('../1sw', x=batches[0], y=batches[1], w_size=batches[2], future=batches[3])
    # print(batches[0])

    # 3) one step timesteps batching
    # from batching_step import DataGeneratorSeq
    #
    # dg = DataGeneratorSeq(data, batch_size=50, num_unroll=300)
    # u_data, u_labels = dg.unroll_batches()
    # u_data, u_labels = np.array(u_data), np.array(u_labels)
    # 4)
#     from batching_gap import batches
#     steps = 15
#     data_offset = 500//30  # 33
#     batch_size = 10
#     u_data, u_labels = batches(data, batch_size=batch_size, steps=steps, offset=data_offset, second_offset=5)
#     print("count",  len(u_data)*len(u_data[0]))
#     print("u_data", len(u_data))  # (300, 10, 2) # maxjor_steps, minor_steps, batchs, time/price
#     print("u_labels", len(u_labels))
#     print("u_labels", u_labels[1][1].shape)
#     # batches = np.stack([u_data, u_labels], axis=1)
#     batches = [u_data, u_labels, data_offset, batch_size]
#     # print(batches)
#
#     data_res = np.array(batches)  # (2, 300, 10, 2) # input/labels, steps, batchs, time/price
#     print(data_res.shape)
#     torch.save(data_res, open('traindata_ya_batch.pt', 'wb'))
#     # print(data_res)
#
#     # from matplotlib import pyplot as plt
#     # plt.plot(range(len(train_data)), a)
#     # plt.show()
#
#
#     # a1 = scaler(d[1])
#     # print(a1)
#     # print(d)
# # data = scale10(data)
# # print(np.mean(data), max(data), min(data))
#
# #
# # # print(data_min, data_max)
# # scale = np.nanstd(data, axis=0)
# # data /= scale
# # print(np.nanstd(data, axis=0))
# # mean = np.nanmean(data, axis=0)
# # data -= mean
# # print(np.nanmean(data, axis=0))
# # print(np.nanstd(data, axis=0))
# # #
# # # print(data)
