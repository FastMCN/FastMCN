import torch
from torch.nn.modules.module import Module
from torch import nn
from torch.nn import functional as F


class TriphaseTanh(Module):
    def forward(self, input):
        input_pow = input.pow(2)
        return input_pow / (input_pow + 16.0) * torch.tanh(input)


class Gelu(Module):
    def forward(self, input):
        return F.gelu(input)


Tanh = nn.Tanh
ReLU = nn.ReLU
