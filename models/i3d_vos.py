#--rgb_sample_path data/Davis/breakdance-flare-rgb.npy
import math
import os

import numpy as np
import torch
from torch.nn import ReplicationPad3d
import matplotlib.pyplot as plt


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D_VOS(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D_VOS, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        # self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.avg_pool = torch.nn.AvgPool3d((2, 4, 4), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        """self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)"""
        self.softmax = torch.nn.Softmax(1)

        #Upsampling Module
        self.side_prep1 = torch.nn.Conv3d(in_channels=64, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.score_dsn1 = torch.nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.upsample1 = torch.nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(4,4,4), stride=(2,2,2),padding=(1,1,1), bias=False)
        self.upsample1_ = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)

        self.side_prep2 = torch.nn.Conv3d(in_channels=192, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.score_dsn2 = torch.nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.upsample2 = torch.nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(4,8,8), stride=(2,4,4),padding=(1,2,2), bias=False)
        self.upsample2_ = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(4,8,8), stride=(2,4,4),padding=(1,2,2), bias=False)

        self.side_prep3 = torch.nn.Conv3d(in_channels=480, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.score_dsn3 = torch.nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.upsample3 = torch.nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(4,16,16), stride=(2,8,8),padding=(1,4,4), bias=False)
        self.upsample3_ = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(4,16,16), stride=(2,8,8), padding=(1,4,4), bias=False)

        self.side_prep4 = torch.nn.Conv3d(in_channels=832, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.score_dsn4 = torch.nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.upsample4 = torch.nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(8,32,32), stride=(4,16,16),padding=(2,8,8), bias=False)
        self.upsample4_ = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(8,32,32), stride=(4,16,16), padding=(2,8,8), bias=False)

        self.fuse = torch.nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, inp): # [18, 3, 16, 112, 112]

        side = []
        side_out = []

        # Preprocessing
        out = self.conv3d_1a_7x7(inp) # [18, 64, 8, 56, 56]

        side_temp1 = self.side_prep1(out) # [18, 16, 8, 56, 56]
        out1 = self.upsample1(side_temp1) # [18, 16, 16, 112, 112]
        side.append(out1)
        out1_ = self.upsample1_(self.score_dsn1(side_temp1)) # [18, 1, 16, 112, 112]
        side_out.append(out1_)

        out = self.maxPool3d_2a_3x3(out) #64x32x56x56
        out = self.conv3d_2b_1x1(out) #64x32x56x56

        out = self.conv3d_2c_3x3(out) # [18, 192, 8, 28, 28]

        side_temp2 = self.side_prep2(out) # [18, 16, 8, 28, 28]
        out2 = self.upsample2(side_temp2) # [18, 16, 16, 112, 112]

        side.append(out2)
        out2_ = self.upsample2_(self.score_dsn2(side_temp2)) # [18, 1, 16, 112, 112]
        side_out.append(out2_)

        out = self.maxPool3d_3a_3x3(out) #192x32x28x28
        out = self.mixed_3b(out) #256x32x28x28
        out = self.mixed_3c(out) # [18, 480, 8, 14, 14]

        side_temp3 = self.side_prep3(out) # [18, 16, 8, 14, 14]
        out3 = self.upsample3(side_temp3)
        side.append(out3)
        out3_ = self.upsample3_(self.score_dsn3(side_temp3))
        side_out.append(out3_)

        out = self.maxPool3d_4a_3x3(out) #480x16x14x14
        out = self.mixed_4b(out) #512x16x14x14
        out = self.mixed_4c(out) #512x16x14x14
        out = self.mixed_4d(out) #512x16x14x14
        out = self.mixed_4e(out) #528x16x14x14
        out = self.mixed_4f(out) #[18, 832, 4, 7, 7]

        side_temp4 = self.side_prep4(out) # [18, 16, 4, 7, 7]
        out4 = self.upsample4(side_temp4)
        side.append(out4)
        out4_ = self.upsample4_(self.score_dsn4(side_temp4))
        side_out.append(out4_)

        out_upsample = torch.cat(side[:], dim=1) # [18, 64, 16, 112, 112]
        out_upsample = self.fuse(out_upsample) # [18, 1, 16, 112, 112]
        side_out.append(out_upsample)

        out = self.maxPool3d_5a_2x2(out) #832x8x7x7
        out = self.mixed_5b(out) #832x8x7x7
        out = self.mixed_5c(out) #1024x8x7x7
        feature = self.avg_pool(out) # [18, 1024, 1, 1, 1]
        # out = self.dropout(out)
        # out = self.conv3d_0c_1x1(out)
        # out = out.squeeze(3)
        # out = out.squeeze(3)
        # out = out.mean(2)
        # out_logits = out
        # out = self.softmax(out_logits)
        return out, feature, side_out
    
        # return side_out

    def upsample_filt(self, size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    def interp_surgery(self, layer):
        m, k, f, h, w = layer.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = self.upsample_filt(h)

        for i in range(m):
            for j in range(f):
                layer.weight[i, i, j, :, :].data.copy_(torch.from_numpy(filt))

        return layer.weight.data

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.ConvTranspose3d):
                m.weight.data.zero_()
                m.weight.data = self.interp_surgery(m)
            elif isinstance(m, torch.nn.Conv3d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
        state_dict = torch.load('models/model_rgb.pth')
        ks = ["conv3d_0c_1x1.conv3d.weight", "conv3d_0c_1x1.conv3d.bias"]
        for k in ks:
            del state_dict[k]
        self.load_state_dict(state_dict, False)

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        load_conv3d(
            state_dict,
            'conv3d_0c_1x1',
            sess,
            os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
            bias=True,
            bn=False)
        self.load_state_dict(state_dict)


    def prime_powers(self, n):
        """
        Compute the factors of a positive integer
        Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
        :param n: int
        :return: set
        """
        factors = set()
        for x in xrange(1, int(math.sqrt(n)) + 1):
            if n % x == 0:
                factors.add(int(x))
                factors.add(int(n // x))
        return sorted(factors)

    def get_grid_dim(self, x):
        """
        Transforms x into product of two integers
        :param x: int
        :return: two ints
        """
        factors = self.prime_powers(x)
        if len(factors) % 2 == 0:
            i = int(len(factors) / 2)
            return factors[i], factors[i - 1]

        i = len(factors) // 2
        return factors[i], factors[i]

    def visualizeActivations(self, out):
        number_frames = out.data.shape[2]
        number_channels = out.data.shape[1]
        rows = out.data.shape[3]
        cols = out.data.shape[4]
        features = out.data.numpy()
        for i in range(0, number_frames):
            features_current = features[:, :, i, :, :]
            features_current = np.reshape(features_current, (number_channels, rows, cols))
            features_current_ = np.transpose(features_current, (1, 2, 0))
            #Normalize
            w_min = np.min(features_current_)
            w_max = np.max(features_current_)

            channels = 1
            num_filters = number_channels

            # get number of grid rows and columns
            grid_r, grid_c = self.get_grid_dim(num_filters)

            # create figure and axes
            fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

            # iterate filters inside every channel
            for l, ax in enumerate(axes.flat):
                # get a single filter
                img_r = (features_current_[:, :, l] - w_min) / (w_max - w_min) * 256
                img_g = (features_current_[:, :, l] - w_min) / (w_max - w_min) * 256
                img_b = (features_current_[:, :, l] - w_min) / (w_max - w_min) * 256
                img = np.dstack((img_r, img_g, img_b)).astype(np.uint8)
                # put it on the grid
                ax.imshow(img)

                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
            # save figure
            plt.savefig('visualize_weights_{}.png'.format(i))

def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        #return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt + '.batch3d.bias'] = torch.from_numpy(beta)
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean)
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var)


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))