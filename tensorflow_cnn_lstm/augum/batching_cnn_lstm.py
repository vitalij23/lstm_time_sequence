import numpy as np


def striding_windows_with_future(arr: list, w_size=5, future=5) -> dict:
    """window slide by 1 form 0 to len-w_size

    :return {x:np.array, y:np.array, x_without_y:np.array}"""

    x = []  # cnn_x
    y = []  # [lstm_x, lstm_y and cnn_y]
    x_without_y = []
    # without least
    for di in range(len(arr) - w_size + 1 - future):
        window = arr[di:di + w_size]
        # print(di, di + w_size, di + w_size + future - 1)
        x.append(window)
        y.append([arr[di + w_size + future - 2], arr[di + w_size + future - 1]])  # for LSTM, for CNN

    # least for prediction
    for di in range(len(arr) - w_size + 1 - future, len(arr) - w_size + 1):
        window = arr[di:di + w_size]
        x_without_y.append(window)
    ret = {"x": np.array(x), "y": np.array(y), "x_without_y": np.array(x_without_y)}

    return ret
