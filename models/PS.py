import torch.nn as nn
from models.PS_parts import *


class PSNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super(PSNet, self).__init__()

        self.inc1 = inconv(9, 12)
        self.down4 = down(12, 24)
        self.down5 = down(24, 48)
        self.down6 = down(48, 96)
        self.down7 = down(96, 96)
        self.tas = MLP_tas(64, 2)


    def forward(self, y):
        y1 = self.inc1(y)
        y2 = self.down4(y1)
        y3 = self.down5(y2)
        y4 = self.down6(y3)
        y5 = self.down7(y4)
        yy = self.tas(y5)

        return yy



class Pred_twistoffset(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Pred_twistoffset, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, out_channel)
        self.activation_2 = nn.Sigmoid()

    def forward(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        x = self.layer3(x)
        output = self.activation_2(x)
        return output
