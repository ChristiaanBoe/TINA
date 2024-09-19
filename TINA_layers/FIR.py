def sinc(x: torch.Tensor):
    """
    Implementation of sinc, i.e. sin(x) / x

    __Warning__: the input is not multiplied by `pi`!
    """
    return torch.where(x == 0, torch.tensor(1., device=x.device, dtype=x.dtype), torch.sin(x) / x)



class FIR_Lowpass_2D_alt(nn.Module):
    def __init__(self, cutoff: float, stride: int = 1, pad: bool = True, zeros: float = 8):
        super(FIR_Lowpass_2D_alt, self).__init__()

        self.cutoff = cutoff
        if self.cutoff < 0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if self.cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.zeros = zeros
        self.half_size = int(zeros / self.cutoff / 2)
        window = torch.hann_window(2 * self.half_size + 1, periodic=False)
        time = torch.arange(-self.half_size, self.half_size + 1)

        if cutoff == 0:
            filter = torch.zeros_like(time)
        else:
            filter = 2 * cutoff * window * sinc(2 * cutoff * math.pi * time)
            # Normalize filter to have sum = 1, otherwise we will have a small leakage
            # of the constant component in the input signal.
            filter /= filter.sum()
        
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, padding=( 0, self.half_size), stride=stride, kernel_size=(1, filter.shape[-1]))
        with torch.no_grad():
            self.conv2d.weight[0, 0] = nn.Parameter(filter.view(1, 1, 1, -1))

    def forward(self, input):

        
        input = input.view(1, 1, 1, -1)


        out = self.conv2d(input)

        out = out.view(1, -1, 1, 1)
        

        return out