import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import random
from torchvision.models.resnet import resnet18


# __all__ = ['resnet18_imagenet', 'resnet18_imagenet_aux', 'resnet18_imagenet_dks', 'resnet18_imagenet_byot']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out
        out = self.relu(out)

        return out


# # origin
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, branch_layers=[], num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         inplanes_head2 = self.inplanes
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         inplanes_head1 = self.inplanes
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
#                                  512 * block.expansion]
#
#         self.branch_layers = branch_layers
#
#         if len(branch_layers) != 0:
#             self.inplanes = inplanes_head2
#             self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2)
#             self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2)
#             self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)
#
#             self.inplanes = inplanes_head1
#             self.layer4_head1 = self._make_layer(block, 512, branch_layers[1][0], stride=2)
#             self.fc_head1 = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation = stride*self.dilation
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x, y=None, loss_type='cross_entropy', feature=False, embedding=False):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         f1 = x
#         x = self.layer2(x)
#         f2 = x
#         x = self.layer3(x)
#         f3 = x
#         x = self.layer4(x)
#         f4 = x
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         embedding0 = x
#         logits = self.fc(x)
#
#         if len(self.branch_layers) != 0:
#             x = self.layer3_head2(f2)
#             x = self.layer4_head2(x)
#             out2 = x
#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#
#             x2 = self.fc_head2(x)
#
#             x = self.layer4_head1(f3)
#             out1 = x
#             x = self.avgpool(x)
#
#             x = x.view(x.size(0), -1)
#
#             x1 = self.fc_head1(x)
#             if feature:
#                 return [logits, x1, x2], [embedding0, f4, out1, out2]
#             else:
#                 return [logits, x1, x2]
#         else:
#             if feature:
#                 return logits, [f1, f2, f3, f4, embedding0]
#             if embedding:
#                 return logits, embedding0
#             else:
#                 return logits


class ResNet(nn.Module):
    def __init__(self, block, layers, branch_layers=[], num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # imagenet
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        inplanes_head2 = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        inplanes_head1 = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)  #imagenet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.branch_layers = branch_layers

        if len(branch_layers) != 0:
            self.inplanes = inplanes_head2
            self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2)
            self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2)
            self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)
            self.inplanes = inplanes_head1
            self.layer4_head1 = self._make_layer(block, 512, branch_layers[1][0], stride=2)
            self.fc_head1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            # bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error !!!")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        stem = x
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        x = self.avgpool(F.relu(feat4))
        x = x.view(x.size(0), -1)
        avg = x
        out = self.fc(x)
        if len(self.branch_layers) != 0:
            x = self.layer3_head2(feat2)
            x = self.layer4_head2(x)
            out2 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x2 = self.fc_head2(x)
            x = self.layer4_head1(feat3)
            out1 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x1 = self.fc_head1(x)
            if feature:
                return [out, x1, x2], [avg, feat4, out1, out2]
            else:
                return [out, x1, x2]
        if feature:
            return out, [feat1, feat2, feat3, feat4, avg]
        else:
            return out


class MixedFusionModule(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1)
        self.bn_low = nn.BatchNorm2d(out_channels)
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1)
        self.bn_high = nn.BatchNorm2d(out_channels)
        self.merge = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.merge_bn = nn.BatchNorm2d(out_channels)

    def forward(self, low_feat, high_feat):
        low_feat = self.pool(low_feat)  # [B, C1, 1, 1]
        low_feat = self.conv_low(low_feat)
        low_feat = self.bn_low(low_feat)
        high_feat = self.conv_high(high_feat)
        high_feat = self.bn_high(high_feat)
        fused = torch.cat([low_feat + high_feat, high_feat], dim=1)
        x = self.merge(fused)
        x = self.merge_bn(x)
        return x


class DTSKDNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DTSKDNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # imagenet
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)  #imagenet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # MFM 模块配置
        self.mfm3 = MixedFusionModule(256, 512, 512)
        self.mfm2 = MixedFusionModule(128, 512, 512)
        self.mfm1 = MixedFusionModule(64, 512, 512)

        # 分支全连接层
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            # bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error !!!")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        stem = x
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        x = self.avgpool(F.relu(feat4))
        feat_high = x
        x = x.view(x.size(0), -1)
        avg = x
        out = self.fc(x)

        f3 = self.mfm3(feat3, feat_high)  # [B, 256, 1, 1]
        f2 = self.mfm2(feat2, f3)  # [B, 256, 1, 1]
        f1 = self.mfm1(feat1, f2)  # [B, 256, 1, 1]
        out1 = self.fc1(f1.view(f1.size(0), -1))  # 分支1
        out2 = self.fc2(f2.view(f2.size(0), -1))  # 分支2
        out3 = self.fc3(f3.view(f3.size(0), -1))  # 分支3

        return out, out3, out2, out1


class DTSKDNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DTSKDNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # imagenet
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)  #imagenet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # MFM 模块配置
        self.mfm3 = MixedFusionModule(1024, 2048, 2048)
        self.mfm2 = MixedFusionModule(512, 2048, 2048)
        self.mfm1 = MixedFusionModule(256, 2048, 2048)

        # 分支全连接层
        self.fc1 = nn.Linear(2048, num_classes)
        self.fc2 = nn.Linear(2048, num_classes)
        self.fc3 = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        stem = x
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        x = self.avgpool(F.relu(feat4))
        feat_high = x
        x = x.view(x.size(0), -1)
        avg = x
        out = self.fc(x)

        f3 = self.mfm3(feat3, feat_high)  # [B, 256, 1, 1]
        f2 = self.mfm2(feat2, f3)  # [B, 256, 1, 1]
        f1 = self.mfm1(feat1, f2)  # [B, 256, 1, 1]
        out1 = self.fc1(f1.view(f1.size(0), -1))  # 分支1
        out2 = self.fc2(f2.view(f2.size(0), -1))  # 分支2
        out3 = self.fc3(f3.view(f3.size(0), -1))  # 分支3

        return out, out3, out2, out1


class Auxiliary_Classifier(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Auxiliary_Classifier, self).__init__()

        layers = [1, 1, 1, 1]
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.block_extractor1 = nn.Sequential(*[self._make_layer(block, 128, layers[1], stride=2),
                                                self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 128 * block.expansion
        self.block_extractor2 = nn.Sequential(*[self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 256 * block.expansion
        self.block_extractor3 = nn.Sequential(*[self._make_layer(block, 512, layers[3], stride=2)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation = stride * self.dilation
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        aux_logits = []
        aux_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            aux_feats.append(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)

            out = getattr(self, 'fc' + str(idx))(out)
            aux_logits.append(out)

        return aux_logits, aux_feats


class ResNet_Final_Auxiliary_Classifer(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet_Final_Auxiliary_Classifer, self).__init__()
        self.conv = conv1x1(512 * block.expansion * 4, 512 * block.expansion)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        sum_fea = torch.cat(x, dim=1)
        out = self.conv(sum_fea)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResNet_Auxiliary(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_Auxiliary, self).__init__()
        self.backbone = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)
        self.auxiliary_classifier = Auxiliary_Classifier(block, layers, num_classes=num_classes,
                                                         zero_init_residual=zero_init_residual)
        self.final_aux_classifier = ResNet_Final_Auxiliary_Classifer(block, num_classes)

    def forward(self, x, lam=0.5, index=None, test=False):
        logits, features = self.backbone(x, is_feat=True)
        aux_logits, aux_feats = self.auxiliary_classifier(features[:-1])
        aux_feats.append(features[-1])
        aux_logits.append(logits)
        if test:
            return aux_logits, aux_feats, 0, 0
        else:
            bs = features[0].size(0)
            if index is None:
                index = torch.arange(math.ceil(bs / 2))

            ensemble_features = [lam * (fea[:bs // 2]) + (1 - lam) * (fea[index]) for fea in aux_feats]
            ensemble_mixup_features = [fea[bs // 2:] for fea in aux_feats]

            ensemle_logits = self.final_aux_classifier(ensemble_features)
            ensemble_mixup_logits = self.final_aux_classifier(ensemble_mixup_features)

        return aux_logits, aux_feats, ensemle_logits, ensemble_mixup_logits


def resnet18_imagenet_dtskd(**kwargs):
    return DTSKDNet18(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_imagenet_dks(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], branch_layers=[[1, 2], [2]], **kwargs)


def resnet18_imagenet_byot(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], branch_layers=[[1, 1], [1]], **kwargs)


def resnet50_imagenet_dtskd(**kwargs):
    return DTSKDNet50(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet18_imagenet_manifold(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_imagenet(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_imagenet_dks(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], branch_layers=[[1, 2], [2]], **kwargs)


def resnet18_imagenet_byot(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], branch_layers=[[1, 1], [1]], **kwargs)


def resnet18_imagenet_aux(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_imagenet(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet34_imagenet_aux(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_imagenet(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50_imagenet_aux(**kwargs):
    return ResNet_Auxiliary(Bottleneck, [3, 4, 6, 3], **kwargs)


if __name__ == '__main__':
    net = resnet18_imagenet(num_classes=1000)
    ckpt = torch.save(net.state_dict(), 'resnet18_imagenet.pth')
    weight = torch.load('resnet18_imagenet.pth', map_location='cpu')
    for name, param in weight.items():
        print(name)
    assert 1 == 0
    x = torch.randn(2, 3, 224, 224)
    y, feature = net(x, feature=True)
    for i in feature:
        print(i.shape)
    from utils import cal_param_size, cal_multi_adds

    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 224, 224)) / 1e6))
