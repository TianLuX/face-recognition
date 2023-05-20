# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models.resnet import resnet50

# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 预训练的resnet50
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.in_channel = 64
        self.res_feature = resnet50(pretrained=True)


    def forward(self, input):
        output = self.res_feature(input)
        return output

# 自己写的resnet50

# bottleneck，残差块，用于构建resnet
class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=[1,1,1],padding=[0,1,0],first=False):
        """
        :param in_channels: 输入维度
        :param out_channels: 输出维度
        :param stride: 第二个卷积步长
        :param identity: 如果输入输出维度相同则直接相加，不同时使用1*1卷积
        """
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )
        # 由于存在维度不一致的情况,所以分情况
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # 卷积核为1
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self,x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = f.relu(out)

        return out

# 自己写的resnet50
class MyResnet50(nn.Module):
    def __init__(self,Bottleneck):
        super(MyResnet50, self).__init__()
        self.in_channels = 64
        # 第一层没有残差块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.in_channels)
        self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(self.in_channels)
        self.sa1 = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.affine = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1000),
        )


    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        flag = True  # 用来判断是否为每个block层的第一层
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.affine(x)
        return x


# Triplet loss损失函数，也可以使用pytorch自带的Triplet loss
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 2.0

    def forward(self, anchor,positive,negative,device):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        pos_dist = f.pairwise_distance(anchor, positive)
        neg_dist = f.pairwise_distance(anchor, negative)
        loss = f.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = f.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                          2))

        return loss_contrastive