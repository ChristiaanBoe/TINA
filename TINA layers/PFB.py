import DFT
import torch
import torch.nn as nn
import numpy as np

class PFB_FIR_DFT(nn.Module):
    def __init__(self, win_coeffs, M, P, expected_input_size):
        super(PFB_FIR_DFT, self).__init__()
        self.win_coeffs = win_coeffs.reshape((M, P)).T
        self.win_coeffs = self.win_coeffs.unsqueeze(0).unsqueeze(1)
        self.win_coeffs = self.win_coeffs.view(P, 1, 1, M)
        self.P = P
        self.M = M
        self.size = expected_input_size
        self.W = int(self.size / self.M / self.P)
        self.Maxsize =  self.M * self.W - self.M
        self.WM = self.M * self.W
        self.FIR = nn.Conv2d(in_channels=self.P, out_channels=self.P, kernel_size=(1, self.M), stride=(1, 1), padding=(0, 0), bias=False, groups=self.P)
        self.FFTlayer = DFTLayer(input_size=self.Maxsize)
        self.FIR.weight = nn.Parameter(self.win_coeffs)
        for param in self.FIR.parameters():
            param.requires_grad = False

    def forward(self, input):
        input = self.FIR(input.view(1, self.WM, 1, self.P).permute(0, 3, 2, 1)[:, :, :, 0:self.WM-1])
        out = self.FFTlayer(input.view(1, self.Maxsize, 1, self.P))
        return out