import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.PS_parts import inconv, down, double_conv
from utils.misc import seg_pool_1d, seg_pool_2d

class MaskEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MaskEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.avg_pool_1 = nn.AdaptiveAvgPool2d((4, 4))  
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)  
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)

        out = self.avg_pool_1(out)
        fea = self.avg_pool_2(out)
        out = out.view(out.size(0),-1,4,4) 
        fea = fea.view(fea.size(0), -1)  
        return out,fea

class VideoEncoder(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super(VideoEncoder, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=True)
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)  
        self.inc1 = inconv(36, 48)
        self.down11 = down(48, 24)
        self.down12 = down(24, 12)
        self.down13 = down(12, 4)

        self.inc2 = inconv(60, 48)
        self.down21 = down(48, 24)
        self.down22 = down(24, 12)
        self.down23 = down(12, 4)

        self.inc3 = inconv(48, 48)
        self.down31 = down(48, 24)
        self.down32 = down(24, 12)
        self.down33 = down(12, 4)


    def forward(self, x, label_tas):
        x = x.transpose(1, 2).reshape(-1, 3, 112, 112)
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        fea = self.avg_pool_1(features).reshape(-1, 96, 512)

        fea = fea.transpose(1, 2)
        fea_re_list = []
        for bs in range(fea.shape[0]):
            v1i0 = int(label_tas[bs][0].item())
            v1i1 = int(label_tas[bs][1].item())
            fea_re_list.append(seg_pool_2d(fea[bs].unsqueeze(0), v1i0, v1i1))
        fea_re = torch.cat(fea_re_list, 0).transpose(1, 2)

        x1 = self.inc1(fea_re[:, 0:36, :])
        x1 = self.down11(x1)
        x1 = self.down12(x1)
        x1 = self.down13(x1)

        x2 = self.inc2(fea_re[:, 36:96, :])
        x2 = self.down21(x2)
        x2 = self.down22(x2)
        x2 = self.down23(x2)

        x3 = self.inc3(fea_re[:, 96:144, :])
        x3 = self.down31(x3)
        x3 = self.down32(x3)
        x3 = self.down33(x3)

        return x1, x2, x3


class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()

    def forward(self, fea, mask_fea):
        output = fea*torch.sigmoid(mask_fea)
        return output


class FeatureFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureFusionModule, self).__init__()

        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

    def forward(self, tensorA, tensorB):

        f1 = self.fc1(tensorA)
        f2 = self.fc2(tensorB)
        fused_features = f1+f2
        return fused_features


class C_channel(nn.Module):
    def __init__(self, in_channel, hidden_dim=512, output_dim=64):
        super(C_channel, self).__init__()
        self.inc = inconv(in_channel, 8)
        self.down1 = double_conv(8, 12)
        self.layer0 = nn.Linear(1024, 512)
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, 64)
        self.activation_1 = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x) 
        x2 = self.down1(x1) 
        x3 = self.activation_1(self.layer0(x2)) 
        x4 = self.activation_1(self.layer1(x3))
        x5 = self.activation_1(self.layer2(x4)) 
        return x5