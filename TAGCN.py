import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from TCN import TemporalConvNet
from layer_norm import layer_normal
from utils import cheb_polynomial
device = torch.device('cuda:0')
def cal_linear_num(layer_num, num_timesteps_input):
    result = num_timesteps_input + 4 * (2**layer_num - 1)
    return result


class TCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, channel_size, layer_num, num_timesteps_input, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TCNBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        # self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        # self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        # #
        # channel_size = [12, 12, 12, 12, 10]
        self.tcn = TemporalConvNet(num_inputs=in_channels, num_channels=channel_size, kernel_size=kernel_size)
        linear_num = cal_linear_num(layer_num, num_timesteps_input)
        self.linear = nn.Linear(linear_num, out_channels)
        self.linear2 = nn.Linear(300, 60)

    # def forward(self, X, Weather, is_weather):
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        X = self.tcn(X.permute(0, 2, 1, 3))
        return X


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.U_1 = torch.randn(60, requires_grad=True).cuda()
        self.U_2 = torch.randn(1, 60, requires_grad=True).cuda()
        self.U_3 = torch.randn(1, requires_grad=True).cuda()
        self.b_e = torch.randn(1, 5, 5, requires_grad=True).cuda()
        self.V_e = torch.randn(5, 5, requires_grad=True).cuda()

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        lhs = torch.matmul(torch.matmul(X.permute(0, 2, 3, 1), self.U_1), self.U_2)
        rhs = torch.matmul(X, self.U_3)
        # product = torch.matmul(lhs.permute(0,2,1), rhs).permute(0,2,1)
        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        # normailzation
        E = E - torch.max(E, 1, keepdim=True)[0]
        exp = torch.exp(E)
        out = exp / torch.sum(exp, 1, keepdim=True)

        return out


def cal_channel_size(layers, timesteps_input):
    channel_size = []
    for i in range(layers - 1):
        channel_size.append(timesteps_input)
    channel_size.append(timesteps_input - 2)
    return channel_size




class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels_1, out_channels_2, num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, num_timesteps_input, cheb_polynomials):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        tcn_layer = 5
        channel_size = cal_channel_size(tcn_layer, num_timesteps_input)
        self.temporal1 = TCNBlock(in_channels=in_channels, out_channels=out_channels, channel_size=channel_size,layer_num=tcn_layer, num_timesteps_input=num_timesteps_input)

        self.Theta1 = nn.Parameter(torch.FloatTensor(60, 300))                      #
        channel_size = cal_channel_size(tcn_layer, num_timesteps_input - 2)
        self.temporal2 = TCNBlock(in_channels=in_channels, out_channels=out_channels, channel_size=channel_size,layer_num=tcn_layer, num_timesteps_input=num_timesteps_input)

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()
        self.cheb_polynomials = cheb_polynomials
        self.conv = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 18), padding=(1, 1))
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    # def forward(self, X, A_hat, Weather):
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # (batch_size, num_of_vertices,
        #  num_of_timesteps, num_of_features) = X.shape

        # global device

        t = self.temporal1(X)
        A = np.array(A_hat.cpu())
        cheb_polynomials = torch.tensor(cheb_polynomial(A, 3), dtype=torch.float32).to(device="cpu")
        outputs = []

        #     graph_signal = t[:, :, :, time_step]
        for time_step in range(5):
            # shape is (batch_size, V, F)
            graph_signal = t[:, :, :, time_step]
            # graph_signal = x[:, :, :, time_step]
            output = torch.zeros(29, 5, 300).to(device)
            for k in range(3):
                T_k = cheb_polynomials[k]
                T_k = T_k.to(device = device)
            # rhs = torch.matmul(T_k, t.permute(1, 0)).permute(1, 0)
                rhs = torch.matmul(T_k, graph_signal).permute(0, 2, 1)
                t2 = F.relu(torch.matmul(rhs, self.Theta1))
                # output = output + torch.matmul(rhs, theta_k)
                output = output + t2
            outputs.append(torch.unsqueeze(output, -1))
        outputs = F.relu(torch.cat(outputs, dim=-1))

        t3 = self.temporal2(outputs.permute(0, 2, 1, 3))

        # a=self.batch_norm(t3)
        return self.batch_norm(t3)

        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, cheb_polynomial):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=60,
                                 spatial_channels=16, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input,cheb_polynomials=cheb_polynomial)
        # self.block2 = STGCNBlock(in_channels=64, out_channels=24,
        #                          spatial_channels=16, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input)
        self.num_timesteps_output = num_timesteps_output
        # self.block1 = STGCNBlock(in_channels=num_features, out_channels_1=16, out_channels_2=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels_1=16, out_channels_2=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=60, out_channels=60)
        # self.fully = nn.Linear(384, num_timesteps_output)
        self.conv = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(5, 10), padding=(0, 0))
        self.fully = nn.Linear(120, 1)


    # def forward(self, A_hat, X, Weather):
    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # out3 = self.block1(X, A_hat, Weather)
        t_attention = self.last_temporal(X)
        # X = torch.einsum("ilj,jm->ilm", [X, t_attention])
        X = torch.matmul(X.reshape(-1, 60, 5), t_attention)
        X = X.reshape(-1, 60, 5, 1)
        out3 = self.block1(X, A_hat)
        # out2 = self.block2(out3, A_hat)
        # out3 = torch.unsqueeze(out3, dim=-1)
        out3 = F.relu(self.conv(out3))
        # out3 = F.relu(self.conv1(out3))
        # out2 = self.block2(out1, A_hat)
        # out3 = self.last_temporal(out2)
        # out3 = layer_normal(out2)
        # out4 = self.fully(out3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)#.reshape((out3.shape[0], out3.shape[1], self.num_timesteps_output, -1)))
        out4 = self.fully(out3)
        return out4


