import numpy as np
import torch
import torch.nn as nn

class IDFTLayer(nn.Module):
    def __init__(self, input_size):
        super(IDFTLayer, self).__init__()
        self.input_size = input_size
        self.IFFTconv = nn.Conv2d(self.input_size, (self.input_size * 2), kernel_size=(1, 1), bias=False)

        """
        F = torch.zeros((input_size, input_size), dtype=torch.complex128)

        for x in range(self.input_size):
          for u in range(self.input_size):
              F[x, u] = torch.exp(-2j * torch.tensor(np.pi) * x * u / self.input_size) / torch.sqrt(torch.tensor(self.input_size, dtype=torch.float64))

        max_abs = torch.max(torch.abs(F))

        # Normalize the DFT matrix between -1 and 1
        F /= max_abs
        F = torch.conj(F)
        #F = torch.transpose(F)
        """
        F = torch.conj(torch.from_NumPy(np.fft.ifft(np.eye(self.input_size))))
        weightsone = torch.unsqueeze(torch.add(F.imag.float(), F.real.float()), -1)
        weightstwo = torch.unsqueeze(torch.add(F.imag.float(), F.real.float()), -1)
        weightsone = torch.unsqueeze(weightsone, -1)
        weightstwo = torch.unsqueeze(weightstwo, -1)
        combined = torch.cat((weightsone, weightstwo), dim=1)
        #weights = torch.unsqueeze(weights, -1)  # Adjust shape for Conv2d

        print(combined.size())

        #print(self.FFTlin.weight.data.shape)
        self.IFFTconv.weight.data = combined
        self.IFFTconv.weight.data.requires_grad = False

        #self.FFTlin..weight.data[] = realweigts.float()





    def forward(self, x):
        # Ensure the input size matches the specified input size
        #if x.size(-1) != self.input_size:
            #raise ValueError(f"Input size must be {self.input_size}, but got {x.size(-1)}")

        # Apply FFT along the last dimension (assuming it's the time dimension)
        output = self.IFFTconv(x)



        return output