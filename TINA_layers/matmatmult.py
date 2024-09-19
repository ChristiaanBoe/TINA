import torch

class matrixmatrixProduct(nn.Module):


  def __init__(self, matrix) -> None:
      super(dotProduct, self).__init__()
      shape = matrix.shape
      self.height, self.width = shape
      
      self.productlayer = nn.Conv2d(self.height, self.height, kernel_size=(1, 1), bias=False)

      self.productlayer.weight.data[: ,:,:] = torch.unsqueeze(torch.unsqueeze(matrix.float(), -1), -1)

      self.productlayer.weight.requires_grad = False  # Set to `True` if you want to fine-tune the weights
  def forward(self, x):
      output = self.FFTconv(x)
      return output