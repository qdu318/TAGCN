# encoding utf-8
'''
@Author: william
@Description:
@time:2020/6/19 16:51
'''
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    相当于一个Residual block
    :param n_inputs: int, 输入通道数
    :param n_outputs: int, 输出通道数
    :param kernel_size: int, 卷积核尺寸
    :param stride: int, 步长，一般为1
    :param dilation: int, 膨胀系数
    :param padding: int, 填充系数
    :param dropout: float, dropout比率
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.c=nn.Conv1d(in_channels=5, out_channels=n_outputs, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.conv1 = weight_norm(nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size= kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.conv1 = weight_norm(nn.Conv1d(in_channels=5, out_channels=5, kernel_size= kernel_size, stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(in_channels=5, out_channels=5, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        #self.downsample = nn.Conv1d(in_channels=5, out_channels=n_outputs, kernel_size=2 * padding + 1, padding=padding)
        self.downsample = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=2 * padding + 1, padding=padding)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化
        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        #x=x.permute(0,3,1,2)
        a=self.conv1(x)
        a=self.chomp1(a)
        a=self.relu1(a)
        a=self.dropout1(a)
        b=self.conv2(a)
        b=self.chomp2(b)
        b=self.relu2(b)
        out=self.dropout2(b)
        #out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。
        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i] if i == 0 else num_channels[i-1]
            # in_channels = out_channels if i == 0 else num_channels[i-1]
            # in_channels = 1
            out_channels = num_channels[0]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)
