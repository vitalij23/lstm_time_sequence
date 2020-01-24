# from __future__ import print_function
# one step - Striding windows in strided timeline
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
from data_augum import striding_windows_reverse

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# TODO: use data as input before future.
# calc loss without test_inpit only future


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.hidden_size = 120
        self.input_size = 1
        self.levels = 4
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.levels)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input: torch.Tensor, future=0):  # 4, 99
        outputs = []
        self.hidden = (
            torch.rand(self.levels, input.size(1), self.hidden_size, dtype=torch.double),  # layers, batch, hidden
            torch.rand(self.levels, input.size(1), self.hidden_size, dtype=torch.double))
        if torch.cuda.is_available():
            self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())

        output = torch.rand(1, input.size(1), 1, dtype=torch.double).cuda()
        # parallel in mini-batch
        for i, input_t in enumerate(input):  # 40 of [99]
            input_t = input_t.view(1, -1, 1)  # [1, 99, 1]
            h_t, self.hidden = self.lstm(input_t, self.hidden)
            output = self.linear(h_t)  # [1, 99, 1]
            # print(output.size())
            if future == 0:
                outputs += [output]  # 40 list of [1, 99, 1]
        for i in range(future):  # if we should predict the future
            h_t, self.hidden = self.lstm(output, self.hidden)
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs).squeeze()  # [40, 1, 99, 1] -> [40, 99]
        return outputs


def main():
    STEPS = 10
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    data: np.array = torch.load('traindata_ya.pt')
    print("batches:", len(data), len(data[0]), type(data), data.shape)
    # [100, 1000] we use 97 inputes for train and 3 for test [97, 999]
    # 100 batches - we learn one function at all batches
    sl = data.shape[0] - data.shape[0] // 3  # test amount
    input = torch.from_numpy(data[:sl, :-1])  # range (-1 1) first sl, without last 1
    # [100, 1000] we use 97 inputes for train and 3 for test [97, 999]
    target = torch.from_numpy(data[:sl, 1:]).squeeze()  # without first 1
    print("train", target.size())

    future = 300
    test_input = data[sl:, :-1]  # second sl, without last
    print("test", len(test_input), len(test_input[0]))
    test_target = data[sl:, 1:]  # second sl, without first

    test_input = striding_windows_reverse(test_input)[:-future]  # without future
    test_target = striding_windows_reverse(test_target)[:-future]  # without future
    test_input = torch.tensor(test_input, dtype=torch.double).view(-1, 1)
    test_target = torch.tensor(test_target, dtype=torch.double).view(-1, 1)

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        input = input.cuda()  # GPU
        target = target.cuda()  # GPU
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    # build the model
    seq = Sequence()
    seq.double()
    seq = seq.to(device)  # GPU
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.3)
    # begin to train
    for i in range(STEPS):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)  # forward - reset state
            # print("out", out.shape)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        print('begin to predict, no need to track gradient here')
        with torch.no_grad():
            pred = seq(test_input, future=future)
            if len(pred.shape) == 1:
                pred = pred.view(-1, 1)
            loss = criterion(pred[future:,:], test_target)
            print('test loss:', loss.item())
            # GPU
            if torch.cuda.is_available():
                pred = pred.cpu()
                input2 = input.cpu()
            y = pred.detach().numpy()
            input2 = input2.detach().numpy()

        y = np.concatenate(y)
        input2 = striding_windows_reverse(input2)
        data2 = striding_windows_reverse(list(data))
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        train_l = len(input2)
        # plt.plot(np.arange(train_l), np.array(input2[:train_l]), 'r', linewidth=2.0)
        plt.plot(np.arange(len(data2)), np.array(data2), 'r', linewidth=2.0)
        plt.plot(np.arange(train_l, train_l + len(y) - future), y[:-future], 'g' + ':', linewidth=2.0)
        plt.plot(np.arange(train_l + len(y) - future, train_l + len(y)), y[-future:], 'b' + ':', linewidth=2.0)

        # plt.show()
        # draw(y[1], 'g')
        # # draw(y[2], 'b')
        # # draw(y[3], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()


if __name__ == '__main__':
   main()
