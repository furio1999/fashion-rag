import torch.nn as nn


class EMASC(nn.Module):
    def __init__(self, 
                 in_channels: list, 
                 out_channels:list, 
                 kernel_size: int=3,
                 padding: int=1, 
                 stride: int=1, 
                 skip_layers: list=[2,3,4,5],
                 type: str='linear'):
        super().__init__()

        if type == 'linear':
            self.conv = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, out_channels):
                self.conv.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=True, padding=1, stride=1))
            self.apply(self._init_weights)
        elif type == 'nonlinear':
            self.conv = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, out_channels):
                adapter = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, bias=True, padding=1, stride=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, bias=True, padding=1, stride=1),
                )
                self.conv.append(adapter)
            
    
    def forward(self, x:list):
        for i in range(len(x)):
            x[i] = self.conv[i](x[i])
        return x
    

    def _init_weights(self, w):
        if isinstance(w, nn.Conv2d):
            w.weight.data.fill_(0.00)
            w.bias.data.fill_(0.00)

"""
UP
torch.Size([8, 128, 512, 384])
torch.Size([8, 128, 256, 192])
torch.Size([8, 256, 128, 96])
torch.Size([8, 512, 64, 48])
"""

"""
DOWN
torch.Size([8, 512, 64, 48])
torch.Size([8, 512, 128, 96])
torch.Size([8, 512, 256, 192])
torch.Size([8, 256, 512, 384])
"""

