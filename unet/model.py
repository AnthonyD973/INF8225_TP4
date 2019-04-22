# TAKEN AND MODIFIED FROM : https://github.com/AakashKT/pytorch-recurrent-ae-siggraph17/blob/b534e990e659fc8fe76f9ff202e17a467a72b4a7/model.py
import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 
import torch


class RecurrentBlock(nn.Module):

	def __init__(self, input_nc, output_nc, downsampling=False, bottleneck=False, upsampling=False):
		super(RecurrentBlock, self).__init__()

		self.input_nc = input_nc
		self.output_nc = output_nc

		self.downsampling = downsampling
		self.upsampling = upsampling
		self.bottleneck = bottleneck

		if self.downsampling:
			self.l1 = nn.Sequential(
					nn.Conv2d(input_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1)
				)
			self.l2 = nn.Sequential(
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
				)
		elif self.upsampling:
			self.l1 = nn.Sequential(
					nn.Upsample(scale_factor=2, mode='nearest'),
					nn.Conv2d(2 * input_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
				)
		elif self.bottleneck:
			self.l1 = nn.Sequential(
					nn.Conv2d(input_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1)
				)
			self.l2 = nn.Sequential(
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
				)

	def forward(self, inp):

		if self.downsampling:
			op1 = self.l1(inp)
			op2 = self.l2(op1)

			return op2
		elif self.upsampling:
			op1 = self.l1(inp)

			return op1
		elif self.bottleneck:
			op1 = self.l1(inp)
			op2 = self.l2(op1)

			return op2



class RecurrentAE(nn.Module):

	def __init__(self, input_nc):
		super(RecurrentAE, self).__init__()

		self.d1 = RecurrentBlock(input_nc=input_nc, output_nc=32, downsampling=True)
		self.d2 = RecurrentBlock(input_nc=32, output_nc=43, downsampling=True)

		self.bottleneck = RecurrentBlock(input_nc=43, output_nc=43, bottleneck=True)

		self.u2 = RecurrentBlock(input_nc=43, output_nc=32, upsampling=True)
		self.u1 = RecurrentBlock(input_nc=32, output_nc=3, upsampling=True)

	def forward(self, inp):
		d1 = func.max_pool2d(input=self.d1(inp), kernel_size=2)
		d2 = func.max_pool2d(input=self.d2(d1), kernel_size=2)

		b = self.bottleneck(d2)

		u2 = self.u2(torch.cat((b, d2), dim=1))
		u1 = self.u1(torch.cat((u2, d1), dim=1))

		return u1

	def reset_hidden(self):
		pass


