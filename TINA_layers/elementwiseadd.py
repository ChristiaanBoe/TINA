import torch

class ElementwiseAdd(nn.Module):
  def __init__(self, matrix) -> None:
      super(ElementwiseAdd, self).__init__()
      shape = matrix.shape

      self.batch, self.channels, self.height, self.width = shape

      self.conv_layer = nn.Conv2d(self.height*self.width, self.height*self.width, bias=True, kernel_size= (1,1), stride= (1, 1), groups= self.width*self.width)
      weightsconv = torch.ones(self.height*self.width, 1, 1, 1).float()

      bias = matrix.view(self.height*self.width)

      self.conv_layer.weight.data = weightsconv
      self.conv_layer.bias.data = bias

      for param in self.conv_layer.parameters():
            param.requires_grad = False


  def forward(self, x):
    shape = x.shape
    batch_size = shape[0]


    input = x.view(batch_size, self.height*self.width, 1, 1)

    out = self.conv_layer(input)

    return out.view(shape)