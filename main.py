import os
import argparse
import pickle as pk
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from TAGCN import STGCN
# from stgcn_origin import STGCN
# from gcn import STGCN
# from TGCN import STGCN
# from GRU import STGCN
# from ASTGCN import STGCN
# from STDN import STGCN
# from LSTM import STGCN

from utils import get_normalized_adj, RMSE, MAE, MAPE
from data_load import Data_load
from logs.logger import Logger
import pandas as pd

num_timesteps_input = 5
num_timesteps_output = 1

epochs = 500
batch_size = 29

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--log_file', type=str, default='run_log')
args = parser.parse_args()

logger = Logger(args.log_file)


# def train_epoch(training_input, training_target, train_weather, batch_size, means, stds):
def train_epoch(training_input, training_target, batch_size, means, stds):
# def train_epoch(training_input, training_target, batch_size, max_value):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    loss_mean = 0.0
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        # X_batch, y_batch, weather_batch = training_input[indices], training_target[indices], train_weather[0]
        X_batch, y_batch = training_input[indices], training_target[indices]
        # X_batch, y_batch, weather_batch = training_input[indices], training_target[indices], train_weather[indices]
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
            # weather_batch = torch.tensor(weather_batch).cuda()
            stds = torch.tensor(stds).cuda()
            means = torch.tensor(means).cuda()
            # max_value = torch.tensor(max_value).cuda()

        # out = net(A_wave, X_batch, weather_batch)
        out = net(A_wave, X_batch)
        # out = net(A_wave, X_batch)

        out = out * stds + means
        y_batch = y_batch * stds + means
        #y_batch=torch.squeeze(y_batch,dim=1)
        # #
        # out = out * max_value
        # y_batch = y_batch * max_value

        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
        # print('loss: ' + str(loss_mean))
    return loss_mean

# def Cal_eval_index(epoch, out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE, max_value):
def Cal_eval_index(epoch, out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE, means, stds):
# def Cal_eval_index(epoch, out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE, max_X, min_X):
    for item in arr:
        # out_index = out[:, :, item, :]
        # val_target_index = val_target[:, :, item, :]
        out_index = out#[:, :, item, :]
        #val_target_index = val_target[:, :, item, :]
        val_target_index = val_target
        out_unnormalized = (out_index * stds + means)  # [85,24,24]
        target_unnormalized = ( out_unnormalized * stds + means)
        val_loss = loss_criterion(out_index, target_unnormalized).to(device="cpu")
        validation_losses.append(np.asscalar(val_loss.detach().numpy()))

        # out_unnormalized = out_index.detach().cpu().numpy() * stds + means
        # target_unnormalized = val_target_index.detach().cpu().numpy() * stds + means

        #         pd.DataFrame(i_start_pred).to_csv("./results_oneto_test120/{}_to_others_pred".format(i) + str(epoch) +".csv",index=None,header=None)
        #         pd.DataFrame(i_start_true).to_csv("./results_oneto_test120/{}_to_others_true".format(i) + str(epoch) +".csv",index=None,header=None)

        # 按类分
        out_unnormalized = (out_index.detach().cpu().numpy() * stds + means)  # [85,24,24]
        target_unnormalized = (val_target_index.detach().cpu().numpy() * stds + means)
        if (epoch % 500 == 0) & (epoch != 0):
            spatial_pre4 = []
            spatial_true4 = []

            pd.DataFrame(spatial_pre4).to_csv("" + str(epoch) + ".csv", index=None,
                                              header=None)
            pd.DataFrame(spatial_true4).to_csv("" + str(epoch) + ".csv", index=None,
                                               header=None)
            print()

        mae = MAE(target_unnormalized, out_unnormalized)
        validation_MAE.append(mae)

        rmse = RMSE(target_unnormalized, out_unnormalized)
        validation_RMSE.append(rmse)

        mape = MAPE(target_unnormalized, out_unnormalized)
        validation_MAPE.append(mape)

    return validation_losses, validation_MAE, validation_RMSE, validation_MAPE


if __name__ == '__main__':
    torch.manual_seed(7)

    # A, max_value, training_input, training_target, val_input, val_target, test_input, test_target = Data_load(num_timesteps_input, num_timesteps_output)

    # A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target, train_weather, Weather_val = Data_load(num_timesteps_input, num_timesteps_output)
    A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target = \
        Data_load(num_timesteps_input, num_timesteps_output)
    # A, max_X, min_X, training_input, training_target, val_input, val_target, test_input, test_target = Data_load(num_timesteps_input, num_timesteps_output)

    torch.cuda.empty_cache()    # free cuda memory
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    if torch.cuda.is_available():
        A_wave = A_wave.cuda()

    net = STGCN(A_wave.shape[0],
                training_input.shape[2],
                num_timesteps_input,
                num_timesteps_output)
    print('number of parameters: ' + str(sum(param.numel() for param in net.parameters())))


    if torch.cuda.is_available():
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_criterion = nn.MSELoss()
    # loss_criterion = nn.SmoothL1Loss()

    training_losses = []
    validation_losses = []
    validation_MAE = []
    validation_RMSE = []
    validation_MAPE = []
    arr = [0]
    for epoch in range(epochs):
        print("epoch: {}/{}".format(epoch, epochs))
        # loss = train_epoch(training_input, training_target, train_weather, batch_size=batch_size, means=means, stds=stds)
        loss = train_epoch(training_input, training_target, batch_size=batch_size, means=means, stds=stds)
        # loss = train_epoch(training_input, training_target, batch_size=batch_size, max_value=max_value)
        training_losses.append(loss)
        torch.cuda.empty_cache()  # free cuda memory
        # Run validation
        with torch.no_grad():
            net.eval()
            if torch.cuda.is_available():
                val_input = val_input.cuda()
                val_target = val_target.cuda()
                # Weather_val = torch.tensor(Weather_val).cuda()
            # out = net(A_wave, val_input, Weather_val)
            out = net(A_wave, val_input)
            # validation_losses, validation_MAE, validation_RMSE, validation_MAPE = Cal_eval_index(epoch, out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE, max_X, min_X)
            validation_losses, validation_MAE, validation_RMSE, validation_MAPE = \
                Cal_eval_index(epoch, out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE, means, stds)
            # validation_losses, validation_MAE, validation_RMSE, validation_MAPE =
            # Cal_eval_index(epoch, out, val_target,arr, validation_losses,validation_MAE,validation_RMSE,validation_MAPE, max_value)
            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Training loss: {}".format(training_losses[-1]))
        for i in range(len(arr)):
            print("time slice:{}, Validation loss:{}, MAE:{}, MAPE:{}, RMSE:{}"
                  .format(arr[i], validation_losses[-1], validation_MAE[-1],
                          validation_MAPE[-1], validation_RMSE[-1],))

        # plt.plot(training_losses, label="training loss")
        # plt.plot(validation_losses, label="validation loss")
        # plt.legend()
        # plt.show()

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_MAE), fd)

        # if (epoch % 10 == 0) & (epoch != 0):
        #     torch.save(net, './checkpoints/params_' + str(epoch))

