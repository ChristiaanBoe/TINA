import torch

class unfolding_layer(nn.Module):
  def __init__(self,size, stride, padding ) -> None:
      super(slidingwindow_layer, self).__init__()
      self.size = size
      self.stride = stride
      self.padding = padding
      self.Slidwind = nn.Conv2d(in_channels=1, out_channels=self.size, kernel_size= (1, self.size), stride=(1,self.stride), padding=(self.padding,self.padding), bias=False, groups = 1)

  def forward(self, input):
    slidwind_output = self.Slidwind(input)
    
    return slidwind_output