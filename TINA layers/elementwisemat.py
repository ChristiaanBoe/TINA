import torch


class ElementwiseMult(nn.Module):
  def __init__(self, matrix) -> None:
      super(ElementwiseMult, self).__init__()
      shape = matrix.shape

      self.batch, self.channels, self.height, self.width = shape

      self.conv_layer = nn.Conv2d(self.height*self.width, self.height*self.width, bias=False, kernel_size= (1,1), stride= (1, 1), groups= self.width*self.width)

      weightsconv = matrix.view(self.height*self.width, 1, 1, 1)


      self.conv_layer.weight.data = weightsconv

  def forward(self, x):
    shape = x.shape
    batch_size = shape[0]


    input = x.view(batch_size, self.height*self.width, 1, 1)

    out = self.conv_layer(input)

    return out.view(shape)