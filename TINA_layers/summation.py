import torch

class Summation(nn.Module):
  def __init__(self, inputwidth) -> None:
      super(Summation, self).__init__()
      self.inputwidth = inputwidth
      self.linear = nn.Linear(inputwidth , 1, bias=False)
      linearweights = torch.ones(1, inputwidth)
      self.linear.weight.data = linearweights


  def forward(self, x):
    shape = x.shape
    
    input = x.view(-1, inputwidth)

    out = self.linear(input)

    return out.view(shape)