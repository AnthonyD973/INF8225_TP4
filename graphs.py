#!/usr/bin/python3
import sys
import os.path
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
from dataset import RandomResizeDataset
from net import Net
from unet.model import RecurrentAE

def load_checkpoint(filename):
	chkpoint = torch.load(filename)

	epoch = chkpoint['epoch']
	model_state = chkpoint['state_dict']
	optimizer_state = chkpoint['optimizer']

	return model_state, optimizer_state, int(epoch)

def test(net, loader, criterion, threshold=None):
    total_loss = 0.0
    total_count = 0
    net.eval()
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(inputs.to(device) - outputs, labels.to(device))

        # print statistics
        total_loss += loss.item()
        total_count += 1

        if threshold is not None and i >= threshold:
            break

    total_loss /= total_count

    return total_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RecurrentAE, SIGGRAPH \'17')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, help='Model chekpoint saving directory')
    parser.add_argument('--names', type=str, help='Comma-separated list of Experiment Name')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mnist_train = torchvision.datasets.FashionMNIST(args.data_dir + "/train", train=True, download=True)
    dataset_train = RandomResizeDataset(mnist_train, ratio_mu=1.0, ratio_sigma=0.25)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=60)
    mnist_test = torchvision.datasets.FashionMNIST(args.data_dir + "/test", train=False, download=True)
    dataset_test = RandomResizeDataset(mnist_test, ratio_mu=1.0, ratio_sigma=0.25)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=60)
    
    net = Net(3, 5, 5).to(torch.double)
    rcae = RecurrentAE(3).to(torch.double)
    #optimiser = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))
    criterion = torch.nn.MSELoss()
    epoch = 0

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
        rcae = torch.nn.DataParallel(rcae)

    names = args.names.split(",")

    all_train_losses = {}
    all_test_losses = {}
    checkpoints = {}
    for name in names:
        checkpoints[name] = sorted([load_checkpoint(x) for x in glob.glob('%s/%s_*.pt' % (args.save_dir, name))], key=lambda x: x[2])
        
        train_losses = []
        test_losses  = []
        for cp in checkpoints[name]:
            model = rcae if name == 'rcae' else net
            model.load_state_dict(cp[0])
            #optimiser.load_state_dict(cp[1])
            epoch = cp[2]

            train_loss = test(model, loader_train, criterion, 10)
            test_loss  = test(model, loader_test,  criterion, 10)

            train_losses.append(train_loss)
            test_losses .append(test_loss)

        all_train_losses[name] = train_losses
        all_test_losses[name] = test_losses
    
    plt.figure(0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss for each network")
    for name in names:
        plt.plot(all_train_losses[name], label='%s_train' % (name))
        plt.plot(all_test_losses[name],  label='%s_test' % (name))
    plt.legend()

    ratios = np.arange(0.4, 2.0, 0.2)
    losses_train = {}
    losses_test = {}
    for name in names:
        cp = checkpoints[name][-1]
        model = rcae if name == 'rcae' else net
        model.load_state_dict(cp[0])
        #optimiser.load_state_dict(cp[1])
        epoch = cp[2]

        losses_train[name] = []
        losses_test[name] = []
        for ratio in ratios:
            ds_train = RandomResizeDataset(mnist_train, ratio_mu=ratio, ratio_sigma=0.0)
            ds_test  = RandomResizeDataset(mnist_test , ratio_mu=ratio, ratio_sigma=0.0)
            ld_train = torch.utils.data.DataLoader(ds_train, batch_size=60)
            ld_test  = torch.utils.data.DataLoader(ds_test , batch_size=60)

            train_loss = test(model, ld_train, criterion, 10)
            test_loss  = test(model, ld_test,  criterion, 10)

            losses_train[name].append(train_loss)
            losses_test[name] .append(test_loss)

    nfig = 1
    plt.figure(nfig)
    plt.xlabel("Upsample Ratio")
    plt.ylabel("Loss")
    plt.title("Loss vs ratio")
    for name in names:
        plt.plot([pow(2, ratio) for ratio in ratios], losses_train[name], label='%s_train' % (name))
        plt.plot([pow(2, ratio) for ratio in ratios], losses_test[name],  label='%s_test' % (name))
    plt.legend()
    nfig += 1

    plt.show()

    #input("Presse Enter key to continue...")