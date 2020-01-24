# from __future__ import print_function
# one step - Striding windows in strided timeline
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import signal
from torchvision.transforms import Compose, Lambda

matplotlib.use('TkAgg')
stop_signal = False
dtype = torch.double  # None #torch.float64 #torch.float16


def signal_handler(signal, frame):
    global stop_signal
    print('You pressed Ctrl+C!')
    stop_signal = True
    # sys.exit(0)


class Sequence(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, levels: int, device: torch.device):
        super(Sequence, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.levels = levels
        self.device = device
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.levels, dropout=0.1)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.lstm2 = nn.LSTM(self.hidden_size//2, self.hidden_size//2, num_layers=self.levels, dropout=0.1)
        self.linear2 = nn.Linear(self.hidden_size//2, 1)
        # self.where_predict = where_predict
        # self.where_hidden = [None] * len(where_predict)  # saved hidden_state for batched training

    def new_hidden(self, batch_size):
        self.hidden1 = (
            torch.rand(self.levels, batch_size, self.hidden_size, dtype=dtype, device=self.device),  # ht # layers, batch, hidden
            torch.rand(self.levels, batch_size, self.hidden_size, dtype=dtype, device=self.device))  # ct
        self.hidden2 = (
            torch.rand(self.levels, batch_size, self.hidden_size//2, dtype=dtype, device=self.device),  # ht # layers, batch, hidden
            torch.rand(self.levels, batch_size, self.hidden_size//2, dtype=dtype, device=self.device))  # ct
        # if torch.cuda.is_available():
        #     self.hidden1 = (self.hidden1[0].cuda(), self.hidden1[1].cuda())
        #     self.hidden2 = (self.hidden2[0].cuda(), self.hidden2[1].cuda())

    def get_c(self):
        return self.hidden[1]

    def detach_hidden(self):
        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        self.hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())

    def forward(self, input: torch.Tensor):  # input- (batch, price/time)
        input_t: torch.Tensor = input.unsqueeze(0)  # [1, 99, 2]


        # The function torch.randn produces a tensor with elements drawn from a Gaussian distribution of zero mean and unit variance. Multiply by sqrt(0.1) to have the desired variance.
        # batch = self.hidden1[0].size()[1]
        # r = (0.1**0.9)*torch.randn(self.levels, batch, self.hidden_size//2, dtype=dtype, device=self.device)
        # self.hidden1 = (self.hidden1[0] + r, self.hidden1[1] + r)
        # self.hidden2 = (self.hidden2[0] + r, self.hidden2[1] + r)

        # h_t1, self.hidden1 = self.lstm1(input_t[:, :, 0].unsqueeze(-1), self.hidden1)
        # h_t2, self.hidden2 = self.lstm2(input_t[:, :, 1].unsqueeze(-1), self.hidden2)
        # print(h_t2.size())
        # h_t = self.linear1(torch.nn.functional.relu(h_t))
        # h_t, self.hidden2 = self.lstm2(h_t, self.hidden2)
        # h_t = torch.cat([h_t1, h_t2], dim=-1)
        # print(h_t.size())
        output, self.hidden1 = self.lstm1(input_t, self.hidden1)  # [1, 99, 1]
        output = self.linear1(torch.nn.functional.relu(output))
        output, self.hidden2 = self.lstm2(output, self.hidden2)
        output = self.linear2(torch.nn.functional.relu(output))  # [1, 99, 1]
        # output = self.linear2(h_t)
        # save hidden for future predict
        # item = input[-1, 1].item()
        # if item in self.where_predict:
        #     ix = np.where(self.where_predict == item)[0].tolist()[-1]
        #     # ix = np.where(self.where_predict == item)[0].item()
        #     self.where_hidden[ix] = self.hidden
        return output


def main():

    signal.signal(signal.SIGINT, signal_handler)

    STEPS = 60
    # batches_bachprop = 10

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data: np.array = torch.load('traindata_ya_batch.pt')
    data_offset = data[2]
    batch_size = data[3]
    print("data:", type(data), data.shape)

    # sl = data.shape[1] - data.shape[1] // 2  # test amount
    # print(sl)

    # # (2, 300, 10, 2) # input/target, steps, batchs, time/price
    # train_input = torch.from_numpy(data[0, :, :]).double()  # range (-1 1) first sl, without last 1
    train_input = data[0]  # range (-1 1) first sl, without last 1
    # print("train_input", train_input.size())

    # target = torch.from_numpy(data[1, :, :, 1]).squeeze().double()  # without first 1
    target = data[1]  # without first 1
    # print("train_train", target.size())

    # test_input = torch.from_numpy(data[0, sl:, :]).double()
    # print("test_input", test_input.size())
    # test_target = torch.from_numpy(data[1, sl:, :, 1]).squeeze().double()  # second sl, without first

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load original
    data_orig: np.array = torch.load('traindata_ya_orig.pt')  # steps, time/price
    print("batches:", len(data_orig), len(data_orig[0]), type(data_orig))
    # build the model
    # where_predict = list(range(130, 500, 100))  # where predict
    # where_predict_dates = data_orig[where_predict, 1]
    # print("where_predict_dates", where_predict_dates)
    seq = Sequence(input_size=1, hidden_size=100, levels=2, device=device)
    seq.to(dtype)
    seq = seq.to(device)  # GPU
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()




    # TRAIN
    # c_save = None
    lr = 0.4 * 1.05
    optimizer = torch.optim.SGD(seq.parameters(), lr=lr, momentum=0.2)
    for s in range(STEPS):
        lr = lr / 1.05
        print("lr", lr)

        for g in optimizer.param_groups:
            g['lr'] = lr
        total_loss = 0.
        iii = 0
        iii_t = 0
        print('STEP: ', s)
        for i, t_in in enumerate(train_input):  # major step +1

            seq.new_hidden(batch_size)
            for ii in range(len(t_in)):  # minor step + offset

                optimizer.zero_grad()

                inp = torch.from_numpy(train_input[i][ii][:, 1]).to(dtype). unsqueeze(-1) #.double()  # batch, time/price
                # print(inp.size())
                tar = torch.from_numpy(target[i][ii][:, 1]).to(dtype) #.double()
                if torch.cuda.is_available():
                    inp = inp.cuda()
                    tar = tar.cuda()

                out: torch.Tensor = seq(inp)  # forward - reset state

                loss: torch.Tensor = criterion(out.squeeze(), tar)
                total_loss += loss.item()

                if ii == (len(t_in) - 1): #or (j == super_step - 1 and ii == 0):
                    # print(i, loss.item())
                    loss.backward()
                    seq.detach_hidden()
                else:
                    loss.backward(retain_graph=True)

                iii += 1
                iii_t += 1
                if iii % (data_offset*10) == 0:
                    print(iii, 'total_loss:', total_loss/iii_t)  # , 'loss:')#, math.exp(cur_loss))
                    total_loss = 0.
                    iii_t = 0

                optimizer.step()
        # print(seq.where_hidden)


        print('begin to predict, no need to track gradient here')
        # TEST
        # test and collect seq.hidden states for where
        res_test_prices = [0]*data_orig.shape[0]
        total_loss = 0.
        with torch.no_grad():
            for i in range(0, data_offset):  # big step  + 1
                seq.new_hidden(1)
                for st in range(i, data_orig.shape[0] - data_offset, data_offset):  # i + offset
                    # ddd[st])
                    # ddd[st + 100]
                    o_place = st + data_offset
                    price_date = torch.from_numpy(data_orig[st])[1].to(dtype).view(1, -1) #.double() # [1,2]
                    next_price = torch.from_numpy(data_orig[o_place])[1].to(dtype).view(1, 1, 1)  # .double()
                    if torch.cuda.is_available():
                        price_date = price_date.cuda()
                        next_price = next_price.cuda()

                    out = seq(price_date)  # steps, batchs, time/price
                    res_test_prices[o_place] = out.item()
                    loss = criterion(out, next_price)
                    total_loss += loss.item()

        print('test loss:', total_loss / (data_orig.shape[0] - data_offset))
        plt.close()
        plt.plot(data_orig[:, 0], data_orig[:, 1], 'r', linewidth=2.0)  # without first
        # future_time = np.arange(2000) / 2000 + 1
        # time = np.concatenate([data[:, 0], future_time])
        # now_time = time[:res_test_prices]
        plt.plot(data_orig[:, 0], res_test_prices, 'g', linewidth=2.0)
        plt.draw()
        plt.pause(0.001)
        if stop_signal:
            break

    # SAVE MODEL
    # plt.pause(99.001)
    torch.save(seq.state_dict(), 'a2.pt')

    exit(0)















    # # PREDICT FUTURE
    # # TODO: apply saved hidden to predict future
    # with torch.no_grad():
    #     for i, w in enumerate(where_predict):
    #         seq.hidden = seq.where_hidden[i]
    #         price_date = torch.from_numpy(data_orig[w, :]).to(torch.float16).unsqueeze(0) #.double() # [1,2]
    #         if torch.cuda.is_available():
    #             price_date = price_date.cuda()
    #         print(price_date.size())
    #         out = seq(price_date)  # steps, batchs, time/price
    #         if i < len(where_predict) - 1:  # one before last
    #             next_price = torch.from_numpy(data_orig[w + 1, :])[1].double()  # [1,2]
    #             print("out", out.shape)
    #             print("next_price", next_price.shape)
    #             loss = criterion(out, next_price)
    #             print('test loss:', loss.item())
    #         # GPU
    #         # if torch.cuda.is_available():
    #         #     pred = pred.cpu()
    #         #     input2 = input.cpu()
    #         # y = pred.detach().numpy()
    #         # input2 = input2.detach().numpy()
    # exit(0)
    #
    #
    # # train = :sl
    # # test = sl:
    # offset = 100  # fixed
    # steps_before = 10
    # steps_after = 5
    #
    # test_pred = []
    # for w in where:
    #     b = w - offset * steps_before
    #     a = w + offset * steps_after
    #     t_p = data[b:a:offset, :]
    #     t_p = torch.from_numpy(t_p).unsqueeze(1).double()
    #     if torch.cuda.is_available():
    #         t_p = t_p.cuda()  # GPU
    #     test_pred.append(t_p)  # 110
    #
    # # print(test_pred.size())
    # yy = []
    # with torch.no_grad():
    #     for t_p in test_pred:  #  = len(where)
    #         print("t_p", t_p[:steps_before, :, :])
    #         pred = seq(t_p[:steps_before, :, :], future=t_p[-steps_after:, :, 0])  # steps, 1, time/price
    #         if len(pred.shape) == 1:
    #             pred = pred.view(-1, 1)
    #         # loss = criterion(pred, test_pred[:, 1].view(-1, 1))
    #         # print('test loss:', loss.item())
    #         # GPU
    #         if torch.cuda.is_available():
    #             pred = pred.cpu()
    #         y = pred.detach().numpy()
    #         yy.append(y)
    #
    #     # y = np.concatenate(y)
    #     # input2 = striding_windows_reverse(input2)
    #     # data2 = striding_windows_reverse(list(data))
    #
    #     # # DRAW THE RESULT
    #
    # plt.figure(figsize=(30, 10))
    # plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    # plt.xlabel('x', fontsize=20)
    # plt.ylabel('y', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # print(yy)
    # # print(input2)
    #
    # plt.plot(data[:, 0], data[:, 1], 'r', linewidth=2.0)
    # # plt.plot(np.arange(len(data2)), np.array(data2), 'g', linewidth=2.0)
    # for i, y in enumerate(yy):  #  = len(where)
    #     a = where[i]+offset * steps_after
    #     future_time = np.arange(2000) / 2000 + 1
    #     time = np.concatenate([data[:, 0], future_time])
    #     now_time = time[where[i]:a:offset]
    #     # print(now_time)
    #     plt.plot(now_time, y, 'g', linewidth=2.0)
    # # plt.plot(np.arange(train_l + len(y) - future, train_l + len(y)), y[-future:], 'b' + ':', linewidth=2.0)
    # plt.savefig('predict%d.pdf' % 1)
    # plt.show()
    # # draw(y[1], 'g')
    # # # draw(y[2], 'b')
    # # # draw(y[3], 'b')
    # plt.savefig('predict%d.pdf' % 1)
    # plt.close()


if __name__ == '__main__':
   main()
