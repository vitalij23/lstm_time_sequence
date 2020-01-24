import numpy as np


def batches(prices: object, batch_size, steps, offset=None, second_offset=1):
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
    ldivo = len_prices / offset
    # if (steps*second_offset + batch_size + offset) < len_prices:
    #     print("Warning Must be: steps*second_offset + batch_size + offset > len_prices")
    if offset is None:
        offset = len_prices // batch_size  # default
    variation = second_offset+1
    x = []  # major unroll + 1
    y = []

    # x_cursors = []
    # y_cursors = []
    aht = 0
    for _ in range(steps):
        for l_o in range(0, offset, second_offset):  # small steps for cursor + 1
            x_sub = []  # sub unroll - + offset
            y_sub = []
            cursor = np.arange(batch_size) + l_o + offset*np.random.randint(0, ldivo-offset)  # [0,1,2,3,4] start point
            while True:  # total steps + offset

                batch_data = [None] * batch_size  # batch
                batch_labels = [None] * batch_size

                y_cursor = cursor + offset + np.random.randint(0, variation, batch_size)
                print(cursor)
                print(y_cursor)
                for i in range(batch_size):
                    c = cursor[i]
                    # print(prices[c])
                    cy = y_cursor[i]
                    batch_data[i] = prices[c]
                    batch_labels[i] = prices[cy]
                # print(batch_data)
                # print(batch_labels)

                # save
                x_sub.append(np.array(batch_data))
                y_sub.append(np.array(batch_labels))

                cursor += offset  # + second_offset
                if cursor[-1] + offset >= len_prices - variation:  # + 1 reset
                    aht += 1
                    break

            x.append(x_sub)
            y.append(y_sub)
            # print("x_sub", np.array(x_sub).shape)
            # print("y_sub", np.array(y_sub).shape)
    print("aht", aht)
    # print("x", np.array(x).shape)
    # print("y", np.array(y).shape)
    # print(x)
    return x, y


if __name__ == '__main__':
    x, y = batches(list(range(100)), 10, steps=45, offset=3, second_offset=2)  # 10 + 45*2 + 3
    # print(y)
