import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, scaling_factor=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.scaler = scaling_factor
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.scaler * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class ResNetwork(nn.Module):

    def __init__(self, block, nb_res_blocks=15):
        super().__init__()
        
        self.inplanes = 64
        # First layer 
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding='same',
                               bias=False)       
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, nb_res_blocks)

        # Last layer
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding='same',
                               bias=False)     
        # Residual 
        self.conv3 = nn.Conv2d(self.inplanes, 2, kernel_size=3, stride=1, padding='same',
                               bias=False)
        
    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = torch.div(c2, 2, rounding_mode="trunc")
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        """Forward pass of the network."""
        iscomplex = False
        if x.shape[-1] == 2:
            x = self.complex_to_chan_dim(x)
            iscomplex = True
        
        out = self.conv1(x)     # first layer output
        x = out

        x = self.layer1(x)

        x = self.conv2(x)       # last layer output
        
        x += out
        
        x = self.conv3(x)

        if iscomplex:
            x = self.chan_complex_to_last_dim(x)

        return x


def resnet_ssdu():
    model = ResNet(BasicBlock)
    return model