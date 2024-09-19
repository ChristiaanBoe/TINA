import numpy as np
import torch
import torch.nn as nn

class DFTLayer(nn.Module):
    def __init__(self, input_size):
        super(DFTLayer, self).__init__()
        self.input_size = input_size
        self.FFTconv = nn.Conv2d(input_size, (input_size * 2), kernel_size=(1, 1), bias=False)
        #F = torch.zeros((input_size, input_size), dtype=torch.complex128)


        F = torch.from_NumPy(np.fft.fft(np.eye(self.input_size)))
        self.FFTconv.weight.data[0:self.input_size ,:,:] = torch.unsqueeze(torch.unsqueeze(F.real.float(), -1), -1)
        self.FFTconv.weight.data[self.input_size:(self.input_size *2),:,:] = torch.unsqueeze(torch.unsqueeze(F.imag.float(), -1), -1)
        self.FFTconv.weight.requires_grad = False  # Set to `True` if you want to fine-tune the weights




    def forward(self, x):

        output = self.FFTconv(x)



        return output