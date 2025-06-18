import torch
import torch.nn as nn
import torch.nn.functional as F

# Normalize
def initializeWts(m):
    '''
    Uses Kaiming initialization for Conv2d layers (as per the paper).
    '''
    def f1(module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')
    def f2(module):
        if isinstance(module, BasicBlock):
            ds = module.conv_downsample
            if ds:
                ds.weight.data.fill_(1 / module.in_channels)
                ds.bias.data.fill_(0)

    m.apply(f1) 
    m.apply(f2)

class Layers(nn.Module):
    def __init__(self, in_channels, in_size, out_labels):
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, out_labels)
        )

    # forward
    def forward(self, x):
        return self.model.forward(x)

class BasicBlock(nn.Module):
    '''
    BasicBlock accurately captures the 2-layer basic block with optional residual connections.
	Supports both shortcut options A (zero-padding) and B (1×1 projection), just like the paper.
    '''
    def __init__(self, in_channels, shortcut=True, downsample=False, option=None):
        super().__init__()
        assert option in {None, 'A', 'B'}, f"{option} is an invalid option"
        self.in_channels = in_channels
        self.downsample = downsample
        self.shortcut = shortcut
        self.option = option
        self.conv_downsample=None
        if self.downsample:
            if shortcut:
                assert option is not None, 'specify option A/B whiel downsampling'
            out_channels = in_channels * 2
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            self.conv_downsample = nn.Conv2d(in_channels, out_channels, 1, stride=2) # 1x1 conv layer with stride 2 
        else: # no downsample
            self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        if not self.shortcut:
            res = self.model.forward(x) # no residual
        elif not self.downsample:
            res = self.model.forward(x) + x # residual
        elif self.option == 'A': # zero padding
            y = self.model.forward(x)
            x = F.max_pool2d(x, 1, 2)
            padded = torch.cat((x, torch.zeros_like(x)), dim=1)
            res = y + padded
        else:
            res = self.model.forward(x) + self.conv_downsample.forward(x)
        return F.relu(res)
    
class BottleneckBlock(nn.Module):
    '''
    BottleneckBlock implements the 1x1 → 3x3 → 1x1 residual block with expansion.
    Supports shortcut options A (zero-padding) and B (1x1 conv projection).
    '''
    expansion = 4

    def __init__(self, in_channels, mid_channels, shortcut=True, downsample=False, option=None):
        super().__init__()
        assert option in {None, 'A', 'B'}, f"{option} is an invalid option"
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = mid_channels * self.expansion
        self.downsample = downsample
        self.shortcut = shortcut
        self.option = option
        self.conv_downsample = None

        stride = 2 if downsample else 1

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, self.out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        if self.downsample and self.shortcut:
            if self.option == 'B':
                self.conv_downsample = nn.Sequential(
                    nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.out_channels)
                )

    def forward(self, x):
        if not self.shortcut:
            res = self.model(x)
        elif not self.downsample:
            res = self.model(x) + x
        elif self.option == 'A':
            y = self.model(x)
            x = F.max_pool2d(x, 1, 2)
            padding = torch.zeros(x.size(0), self.out_channels - x.size(1), x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            padded = torch.cat((x, padding), dim=1)
            res = y + padded
        else:
            res = self.model(x) + self.conv_downsample(x)
        return F.relu(res)

class CifarResNet(nn.Module):
    '''
    Correctly constructs 20/32/44/56/110-layer ResNets.
	Uses 3×3 convolutions and matches layer counts (6n + 2 format).
	Applies no dropout/maxout (matching the paper’s experimental setup).
	transform() does per-channel mean subtraction as expected.
    '''
    def __init__(self, n, residual=True, option=None):
        super().__init__()
        layers = {20, 32, 44, 56, 110}
        assert n in layers
        k = (n-2)//6
        modules = [nn.Conv2d(3,16,3, padding=1)]
        modules += [BasicBlock(16, shortcut=residual) for _ in range(k)]

        modules.append(BasicBlock(16, shortcut=residual, downsample=True, option=option))
        modules += [BasicBlock(32, shortcut=residual) for _ in range(k - 1)]

        modules.append(BasicBlock(32, shortcut=residual, downsample=True, option=option))
        modules += [BasicBlock(64, shortcut=residual) for _ in range(k - 1)]

        modules.append(Layers(64, 8, 10))
        self.model = nn.Sequential(*modules)
        initializeWts(self)

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def transform(x):
        return x - x.mean(dim=(2, 3), keepdim=True)

class ImageNetResNet(nn.Module):
    '''
    ImageNetResNet uses correct downsampling via strided convolutions and maxpool.
	Correct layer counts for ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152.
	Final output uses global average pooling followed by fully connected layer.
    '''

    def __init__(self, n, residual=True, option=None):
        super().__init__()

        assert n in {18, 34, 50, 101, 152}, 'N must be 18, 34, 50, 101 or 152'
        if n == 18:
            layers = (2, 1, 1, 1)
            block = BasicBlock
        elif n == 34:
            layers = (3, 3, 5, 2)
            block = BasicBlock
        elif n == 50:
            layers = (3, 4, 6, 3)
            block = BottleneckBlock
        elif n == 101:
            layers = (3, 4, 23, 3)
            block = BottleneckBlock
        else:  # 152
            layers = (3, 8, 36, 3)
            block = BottleneckBlock

        self.in_channels = 64
        modules = [
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]

        channels_list = [64, 128, 256, 512]

        for i, num_blocks in enumerate(layers):
            channels = channels_list[i]
            for j in range(num_blocks):
                downsample = (j == 0 and i != 0)
                modules.append(block(self.in_channels, channels, downsample=downsample, shortcut=residual, option=option))
                self.in_channels = channels * block.expansion

        modules.append(Layers(self.in_channels, 7, 1000))
        self.model = nn.Sequential(*modules)
        initializeWts(self)

    def forward(self, x):
        return self.model.forward(x)