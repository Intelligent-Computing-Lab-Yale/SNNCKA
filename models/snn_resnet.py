import random
from models.layers import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, mode='normal'):
        super(PreActBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.conv2 = SeqToANNContainer(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.spike = LIFSpike()
        self.mode = mode

        if (stride != 1 or inplanes != planes) and self.mode != 'none':
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(tdLayer(nn.AvgPool2d(stride)),
                                            SeqToANNContainer(conv1x1(inplanes, planes)),
                                            norm_layer(planes))

    def forward(self, x):
        out = self.spike(x)
        residual = self.downsample(out) if hasattr(self, 'downsample') else x
        out = self.bn1(self.conv1(out))
        out = self.bn2(self.conv2(self.spike(out)))
        return out if self.mode == 'none' else out + residual


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, width_mult=1, in_c=3, mode='normal'):
        super(ResNet, self).__init__()

        global norm_layer
        norm_layer = tdBatchNorm

        self.inplanes = 16 * width_mult

        self.conv1 = SeqToANNContainer(nn.Conv2d(in_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn = tdBatchNorm(self.inplanes)

        self.layer1 = self._make_layer(block, self.inplanes, layers[0], mode=mode)
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], stride=2, mode=mode)
        self.layer3 = self._make_layer(block, self.inplanes * 2, layers[2], stride=2, mode=mode)
        self.spike = LIFSpike()
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1 = SeqToANNContainer(nn.Linear(self.inplanes, num_classes))
        self.T = 4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, mode='normal'):

        layers = []
        layers.append(block(self.inplanes, planes, stride, mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, mode=mode))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spike(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x

    def forward(self, x):
        if len(x.shape) == 4:
            x = add_dimention(x, self.T)
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = {}
        model.load_state_dict(state_dict)
    return model



def resnet20(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet20', PreActBasicBlock, [3, 3, 3], pretrained, progress, **kwargs)


def resnet38(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet20', PreActBasicBlock, [6, 6, 6], pretrained, progress, **kwargs)


def resnet56(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet56', PreActBasicBlock, [9, 9, 9], pretrained, progress, **kwargs)


def resnet110(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet110', PreActBasicBlock, [18, 18, 18], pretrained, progress, **kwargs)


def resnet164(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet164', PreActBasicBlock, [27, 27, 27], pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = resnet20(num_classes=10, width_mult=1, mode='none')
    model.T = 4
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    y.mean().backward()