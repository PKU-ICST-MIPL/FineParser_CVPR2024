import torch.nn as nn
import torch
from .i3d import I3D
import logging

class I3D_backbone(nn.Module):
    def __init__(self, I3D_class):
        super(I3D_backbone, self).__init__()
        print('Using I3D backbone')
        self.backbone = I3D(num_classes=I3D_class, modality='rgb', dropout_prob=0.5)
        
    def load_pretrain(self, I3D_ckpt_path):
        state_dict = torch.load('./models/model_rgb.pth')
        ks = ["conv3d_0c_1x1.conv3d.weight", "conv3d_0c_1x1.conv3d.bias"]
        for k in ks:
            del state_dict[k]
        self.backbone.load_state_dict(state_dict)
        print('loading ckpt done')

    def forward(self, video_1, video_2):

        total_video = torch.cat((video_1, video_2), 0)
        start_idx = list(range(0, 90, 10))
        video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])
        total_feamap, total_feature = self.backbone(video_pack)
        Nt, C, T, H, W = total_feamap.size()

        total_feature = total_feature.reshape(len(start_idx), len(total_video), -1).transpose(0, 1)
        total_feamap = total_feamap.reshape(len(start_idx), len(total_video), C, T, H, W).transpose(0, 1)

        com_feature_12 = torch.cat(
            (total_feature[:total_feature.shape[0] // 2], total_feature[total_feature.shape[0] // 2:]), 2)
        com_feamap_12 = torch.cat(
            (total_feamap[:total_feamap.shape[0] // 2], total_feamap[total_feamap.shape[0] // 2:]), 2)
        return com_feature_12, com_feamap_12