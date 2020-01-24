# from __future__ import print_function
# one step - one time in time series for many lines
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import signal

matplotlib.use('TkAgg')
stop_signal = False
dtype = torch.double  # None #torch.float64 #torch.float16


def signal_handler(signal, frame):
    global stop_signal
    print('You pressed Ctrl+C!')
    stop_signal = True
    # sys.exit(0)


class Sequence(nn.Module):
    def __init__(self, input_size: int, device: torch.device):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size, 151)
        self.lstm2 = nn.LSTMCell(151, 251)
        self.lstm3 = nn.LSTMCell(251, 51)
        # self.linear1 = nn.Linear(151, 40)
        self.linear2 = nn.Linear(51, 2)  # price, volume
        self.device = device

    def forward(self, input, future = 0):  # 97, 999
        #
        outputs = []
        # batch_size = input.size(0)
        batch_size = 1
        h_t = torch.zeros(batch_size, 151, dtype=torch.double, device=self.device)  # [97, 51] # batch_size, hidden_size
        c_t = torch.zeros(batch_size, 151, dtype=torch.double, device=self.device)  # [97, 51]
        h_t2 = torch.zeros(batch_size, 251, dtype=torch.double, device=self.device)
        c_t2 = torch.zeros(batch_size, 251, dtype=torch.double, device=self.device)
        h_t3 = torch.zeros(batch_size, 51, dtype=torch.double, device=self.device)
        c_t3 = torch.zeros(batch_size, 51, dtype=torch.double, device=self.device)
        # print(input.size())

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):  # [97, 1]
            input_t = input_t.view(1, -1)
            # print(input_t.size())  # [97, 1]
            # print(h_t.size())
            # print(c_t.size())
            # exit(0)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear2(h_t3)  # [97, 1]
            outputs += [output]  # 999 list of [97, 1]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear2(h_t3)  # [97, 1]
            outputs += [output]
        outputs = torch.stack(outputs).squeeze()  # [97, 999, 1] -> [97, 999]
        # print(outputs.size())

        return outputs


if __name__ == '__main__':
    STEPS = 4

    signal.signal(signal.SIGINT, signal_handler)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    # data = torch.load('traindata.pt')  # N=100, L=1000
    data = np.load('123.npy', mmap_mode=None)
    print(data.shape)  # (2, 500) # price/value, steps
    test_st = 600  # test
    train_st = data.shape[1] - test_st  # train
    input = torch.from_numpy(data[:, :-1 + train_st])
    print(input.shape)
    target = torch.from_numpy(np.transpose(data[:, 1:train_st]))  # 2,3
    print(target.shape)
    # test_input = torch.from_numpy(data[:, :-1])  # first 3 lines
    # test_target = torch.from_numpy(np.transpose(data[:, 1:]))
    #CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # build the model
    seq = Sequence(data.shape[0], device)
    seq.double()
    criterion = nn.MSELoss()
    # CUDA
    if torch.cuda.is_available():
        input = input.cuda()  # GPU
        target = target.cuda()  # GPU
        # test_input = test_input.cuda()
        # test_target = test_target.cuda()
        seq = seq.cuda()
        criterion = criterion.cuda()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.5)
    #begin to train
    lr = 0.5 * 1.6
    for i in range(STEPS):
        print('STEP: ', i)
        lr = lr / 1.6
        print("lr", lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            # print("out", out.size())
            # print("target", target.size())
            loss = criterion(out[:, 0], target[:, 0])
            # print(loss.item())
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
    with torch.no_grad():
        future = train_st
        # print(input.size())
        pred = seq(input, future=test_st)
        # print(pred.size())
        # print(pred[:-future, :].size())
        # print(target.size())
        # print(pred[:t_st, 0].size())
        loss = criterion(pred[1:train_st, 0], target[:, 0])
        print('test loss:', loss.item())
        y = pred.detach()
        if torch.cuda.is_available():
            y = y.cpu()
        y = y.numpy()
    # draw the result
    plt.close()
    # plt.figure(figsize=(20,5))
    # plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    # plt.xlabel('x', fontsize=20)
    # plt.ylabel('y', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.plot(np.arange(data.shape[1]), data[0, :], 'r', linewidth=2.0)
    plt.plot(np.arange(1, train_st+1), y[:train_st, 0], 'b', linewidth=2.0)
    print(train_st + 1, data.shape[1])
    print(y.shape[0], train_st)
    plt.plot(np.arange(train_st + 1, data.shape[1]), y[train_st:, 0], 'g', linewidth=2.0)

    # def draw(yi, color):
    #     plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
    #     plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)
    # draw(y[:, 0], 'r')
    # draw(y[1], 'g')
    # draw(y[2], 'b')
    plt.draw()
    plt.show()
    # plt.pause(0.0001)
    # if stop_signal:
    #     break
    # plt.pause(99)
        # plt.savefig('predict%d.pdf' % i)
        # plt.close()
