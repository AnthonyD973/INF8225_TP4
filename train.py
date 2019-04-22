#!/usr/bin/python3
import sys
import os.path
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
from dataset import RandomResizeDataset
from net import Net

parser = argparse.ArgumentParser(description='RecurrentAE, SIGGRAPH \'17')
parser.add_argument('--data_dir', type=str, help='Data directory')
parser.add_argument('--save_dir', type=str, help='Model chekpoint saving directory')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
parser.add_argument('--name', type=str, help='Experiment Name')
parser.add_argument('--epochs', type=int, help='Number of epochs to train')

args = parser.parse_args()

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def save_checkpoint(state, filename):
	torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
	chkpoint = torch.load(filename)

	epoch = chkpoint['epoch']
	model.load_state_dict(chkpoint['state_dict'])
	optimizer.load_state_dict(chkpoint['optimizer'])

	return model, optimizer, int(epoch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_train = RandomResizeDataset(torchvision.datasets.MNIST(args.data_dir + "/train", train=True, download=True), ratio_mu=1.0, ratio_sigma=0.25)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=60)

dataset_test = RandomResizeDataset(torchvision.datasets.MNIST(args.data_dir + "/test", train=False, download=True), ratio_mu=1.0, ratio_sigma=0.25)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=60)

net = Net(3, 64, 5, 5).to(torch.double)
optimiser = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))
criterion = torch.nn.MSELoss()
epoch = 0

if torch.cuda.is_available():
    net = torch.nn.DataParallel(net)

if len(args.checkpoint) > 0 and os.path.isfile(args.checkpoint):
    print("Loading from \"%s\"..." % (args.checkpoint))
    #net.load_state_dict(torch.load(args.checkpoint))
    net, optimiser, epoch = load_checkpoint(args.checkpoint, net, optimiser)
    print("Loaded")

def train(net, loader, optimizer, criterion, epoch):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(inputs.to(device) - outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 300 == 299:    # print every 50 mini-batches
            print('Train [%d, %5d] loss: %.8f' %
                (epoch + 1, i + 1, running_loss / 300))
            running_loss = 0.0
            if args.save_dir is not None:
                print("Saving to \"%s/%s_%s.pt\"..." % (args.save_dir, args.name, epoch+1))
                save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict':net.state_dict(),
                        'optimizer':optimizer.state_dict(),
                    }, '%s/%s_%s.pt' % (args.save_dir, args.name, epoch+1))
                print("Saved")

def test(net, loader, criterion, epoch):
    running_loss = 0.0
    count = 0
    net.eval()
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(inputs.to(device) - outputs, labels.to(device))

        # print statistics
        running_loss += loss.item()
        count += 1
        if i % 50 == 49:    # print every 50 mini-batches
            print('Test  [%d, %5d] loss: %.8f' %
                (epoch + 1, i + 1, running_loss / count))
            running_loss = 0.0
            count = 0
    if count is not 0:
        print('Test  [%d, %5d] loss: %.8f' %
            (epoch + 1, i + 1, running_loss / count))


for epoch in range(args.epochs):
    train(net, loader_train, optimiser, criterion, epoch)
    test (net, loader_test, criterion, epoch)

    net.eval()
    it = iter(loader_test)
    for i in range(10):
        data, label = next(it)
        plt.figure(0)
        show(torchvision.utils.make_grid(data[1,:].cpu()))
        plt.figure(1)
        show(torchvision.utils.make_grid((data[1,:].cpu() - net(data)[1,:].cpu())).cpu().detach())
        plt.figure(2)
        show(torchvision.utils.make_grid(label[1,:].cpu().detach()))
        plt.pause(2)

if args.save_dir is not None:
    print("Saving to \"%s/%s_%s.pt\"..." % (args.save_dir, args.name, args.epochs))
    save_checkpoint({
            'epoch': args.epochs,
            'state_dict':net.state_dict(),
            'optimizer':optimiser.state_dict(),
        }, '%s/%s_%s.pt' % (args.save_dir, args.name, args.epochs))
    print("Saved")

input("Presse Enter key to continue...")