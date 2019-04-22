import torch
import torchvision
import sys

class Net(torch.nn.modules.Module):
    def __init__(self, n_channels, n_layers, kernel_size = 3):
        super(Net, self).__init__()

        self.n_channels = n_channels
        self.n_layers = n_layers

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 32, kernel_size),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(2)
        )

        self.unfold = torch.nn.Sequential(torch.nn.Unfold(1), torch.nn.LeakyReLU(negative_slope=0.1))

        self.re = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=0.4)
        self.leakyRecurrent = torch.nn.LeakyReLU(negative_slope=0.1)
        self.rd = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=0.4)

        self.fold = torch.nn.Fold

        self.deconv = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ConvTranspose2d(64, 32, kernel_size),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ConvTranspose2d(32, n_channels, kernel_size),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, input):
        batch_size, n_channels, height, width = input.size()
        input = self.conv(input)
        unfolded = self.unfold(input).transpose_(1, 2)
        # input = input.transpose_(1, 3).transpose_(1, 2).contiguous().view(batch_size, -1, n_channels)
        out1, state1 = self.re(unfolded)
        out1 = self.leakyRecurrent(out1)
        out2, _ = self.rd(out1.view(batch_size, -1, 2, 64).sum(2).div(2).squeeze(2), state1) # torch.zeros(batch_size, height * width, self.n_hidden, dtype=out1.dtype, device=out1.device)
        out2 = out2.view(batch_size, -1, 2, 64).sum(2).div(2).squeeze(2).transpose_(1, 2)
        folded = self.fold(input.size()[2:], 1)(out2)
        out = self.deconv(folded)
        #out = self.fc(out2.contiguous().view(-1, self.n_hidden)).view(batch_size, height, width, n_channels).transpose_(1, 3)
        return out
