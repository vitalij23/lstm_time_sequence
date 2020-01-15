import numpy as np


class DataGeneratorSeq(object):

    def __init__(self, prices, batch_size, num_unroll):  # [100] 5,5
        """

        :param prices: list of any objects
        :param batch_size:
        :param num_unroll: count of batches and (count of prices to be placed) - was
        """
        self._prices = prices
        #self._prices_length = len(self._prices) - num_unroll  # 95
        self._prices_length = len(self._prices) - 5  # - batch_size  # 95
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size  # 19
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]  # [0, 19, 19*2=38, 57, 76]

    def next_batch(self):

        # batch_data = np.zeros(self._batch_size, dtype=np.float32)
        # batch_labels = np.zeros(self._batch_size, dtype=np.float32)
        batch_data = [None] * self._batch_size
        batch_labels = [None] * self._batch_size

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                self._cursor[b] = self._cursor[(b - 1)]+2
                # print("BAD", b, self._cursor)
            batch_data[b] = self._prices[self._cursor[b]]  # [0, 19, 38, 57, 76]
            # numbers inside [0, 19, 38, 57, 76]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]  # cursor + 0, 4 random
            self._cursor[b] = (self._cursor[b] + np.random.randint(0, 2)) #% self._prices_length  # increase [0, 19, 38, 57, 76] by 1
            print(self._cursor)
        return batch_data, batch_labels

    def unroll_batches(self) -> (list, list):

        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))


# import random
#
#
# def my(prices, batch_size, num_unroll):
#     """
#
#     :param prices: one dimension
#     :param batch_size: examples in step before gradient optimization
#     :param num_unroll: steps in epoch
#     :return:
#     """
#     prices_length = len(prices) - num_unroll
#     segments = (len(prices) // batch_size) - 1
#     print(segments)  # 19
#     cursor = [offset * segments for offset in range(batch_size)]
#     print(cursor)  # [0, 19, 38, 57, 76]
#     x = []
#     y = []
#     for i in range(num_unroll):  # 50
#         batch_data = np.zeros(batch_size, dtype=np.float32)
#         batch_labels = np.zeros(batch_size, dtype=np.float32)
#
#         # c + num_unroll // batch_size
#
#         xl = [cursor[b] for b in range(batch_size)]  # 5
#         x.append([c + i for c in cursor])
#         # yl = [c + i + random.randint(0, segments-i) for c in cursor]
#         print(xl)
#         # print(yl)
#         # y.append(yl)
#     # print(np.array(pp))
#     # print(pp2)
#     return x, y
#
#
# # my(list(range(100)), 5, 50)
if __name__ == '__main__':
    dg = DataGeneratorSeq(list(range(1000)), 10, 300)
    u_data, u_labels = dg.unroll_batches()

    # for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
    #     print('\n\nUnrolled index %d' % ui)
    #     dat_ind = dat
    #     lbl_ind = lbl
    #     print('\tInputs: ', dat)
    #     print('\tOutput:', lbl)
