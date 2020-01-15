import numpy as np


def batches(prices, batch_size, steps, offset=None, second_offset=1):
    """
    sliding window = batch
    offset = len(prices) / batch_size - may be any

    :param prices: list of any objects
    :param batch_size: examples in step before gradient optimization
    :param steps:
    :param offset:
    :param second_offset: gap inside batch
    :return:
    """

    len_prices = len(prices)
    if (steps*second_offset + batch_size + offset) < len_prices:
        print("Warning Must be: steps*second_offset + batch_size + offset > len_prices")
    if offset is None:
        offset = len_prices // batch_size  # default
    variation = 2
    x = []
    y = []
    cursor = np.arange(batch_size)  # [0,1,2,3,4] start point
    for i in range(steps):  # 50

        batch_data = [None] * batch_size
        batch_labels = [None] * batch_size

        y_cursor = cursor + offset + np.random.randint(0,2,batch_size)
        # print(y_cursor)
        for i in range(batch_size):
            c = cursor[i]
            # print(prices[c])
            cy = y_cursor[i]
            batch_data[i] = prices[c]
            batch_labels[i] = prices[cy]

        # save
        x.append(batch_data)
        y.append(batch_labels)

        cursor += second_offset
        if cursor[-1] + offset >= len_prices + 1 - variation:  # reset
            cursor = np.arange(batch_size)
    return np.array(x), np.array(y)


if __name__ == '__main__':
    x, y = batches(list(range(100)), 10, steps=45, offset=3, second_offset=2)  # 10 + 45*2 + 3
    # print(y)
