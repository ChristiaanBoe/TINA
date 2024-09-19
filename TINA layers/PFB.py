import DFT
import torch
import torch.nn as nn
import numpy as np

class DFTLayer(nn.Module):
    def __init__(self, input_size):
        super(DFTLayer, self).__init__()
        self.input_size = input_size
        self.FFTconv = nn.Conv2d(input_size, input_size * 2, kernel_size=(1, 1), bias=False)

        F = torch.from_numpy(np.fft.fft(np.eye(self.input_size)))
        self.FFTconv.weight.data[0:self.input_size, :, :, :] = torch.unsqueeze(torch.unsqueeze(F.real.float(), -1), -1)
        self.FFTconv.weight.data[self.input_size:(self.input_size * 2), :, :, :] = torch.unsqueeze(torch.unsqueeze(F.imag.float(), -1), -1)

        self.FFTconv.weight.data.requires_grad = False

    def forward(self, x):
        output = self.FFTconv(x)
        return output





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
        self.Maxsize = self.M * self.W - self.M
        self.WM = self.M * self.W

        self.FIR = nn.Conv2d(in_channels=self.P, out_channels=self.P, kernel_size=(1, self.M), stride=(1, 1), padding=(0, 0), bias=False, groups=self.P)
        self.FFTlayer = DFTLayer(input_size=self.P)
        self.FIR.weight = nn.Parameter(self.win_coeffs)

    def forward(self, input):
        input = input.view(input.shape[0], self.WM, 1, self.P).permute(0, 3, 2, 1)[:, :, :, 0:self.WM-1]
        input = self.FIR(input)
        input = self.FFTlayer(input)
        return input
