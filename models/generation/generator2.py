# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#08.09 change pad

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.generation.base_network import BaseNetwork

import torch.nn.utils.spectral_norm as spectral_norm

class depth_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(depth_separable_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class sc_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(sc_conv, self).__init__()
        self.single_channel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1
        )

    def forward(self, input):
        out = self.single_channel_conv(input)
        return out


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='replicate',
                 activation='elu', norm='none', sc=False, sn=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sc:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = sc_conv(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = depth_separable_conv(in_channels, out_channels, kernel_size, stride, padding=0,
                                                    dilation=dilation)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_in):
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        return x


from ..spade.networks.architecture import SPADEResnetBlock2
from .architecture import ResnetBlock_adain

class PSGIM_network(BaseNetwork):
    def __init__(self, input_dim, label_nc, style_dim=256, nf=64):
        super().__init__()
        self.G_middle_0 = SPADEResnetBlock2(input_dim, 8 * nf, norm_G='spadebatch3x3', semantic_nc=1)
        self.G_middle_1 = SPADEResnetBlock2(8 * nf, 4 * nf, norm_G='spadebatch3x3', semantic_nc=1)
        self.up_0 = SPADEResnetBlock2(4*nf, 2*nf, norm_G='spadebatch3x3', semantic_nc=1)

        self.atrous1 = GatedConv2d(8 * nf, 8 * nf, 3, 1, padding=2, dilation=2, sc=True)
        self.atrous2 = GatedConv2d(8 * nf, 8 * nf, 3, 1, padding=4, dilation=4, sc=True)
        self.atrous3 = GatedConv2d(8 * nf, 8 * nf, 3, 1, padding=8, dilation=8, sc=True)
        self.atrous4 = GatedConv2d(8 * nf, 8 * nf, 3, 1, padding=16, dilation=16, sc=True)

        self.atrous1_conv = GatedConv2d(4 * nf, 4 * nf, 3, 1, padding=1, dilation=1, sc=True)
        self.atrous2_conv = GatedConv2d(4 * nf, 4 * nf, 3, 1, padding=1, dilation=1, sc=True)
        self.atrous3_conv = GatedConv2d(4 * nf, 4 * nf, 3, 1, padding=1, dilation=1, sc=True)
        self.atrous4_conv = GatedConv2d(4 * nf, 4 * nf, 3, 1, padding=1, dilation=1, sc=True)

        self.atrous1_conv2 = GatedConv2d(2 * nf, 2 * nf, 3, 1, padding=1, dilation=1, sc=True)
        self.atrous2_conv2 = GatedConv2d(2 * nf, 2 * nf, 3, 1, padding=1, dilation=1, sc=True)
        self.atrous3_conv2 = GatedConv2d(2 * nf, 2 * nf, 3, 1, padding=1, dilation=1, sc=True)
        self.atrous4_conv2 = GatedConv2d(2 * nf, 2 * nf, 3, 1, padding=1, dilation=1, sc=True)

        self.conv_img = nn.Conv2d(8 * nf, 3, 3, padding=1)
        self.conv_img2 = nn.Conv2d(4 * nf, 3, 3, padding=1)
        self.conv_img3 = nn.Conv2d(2*nf, 3, 3, padding=1)

        self.style_encoder = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
                                           nn.ReLU(),
                                           nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                           nn.ReLU(),
                                           nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
                                           nn.ReLU(),
                                           nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(256, style_dim, 1, 1, 0))
        self.style_transfer = ResnetBlock_adain(input_dim, input_dim)
        adain_num = self.get_num_adain_params(self.style_transfer)
        print('adain num:', adain_num, ' ,style dim:', style_dim)
        self.mlp = nn.Sequential(nn.Linear(style_dim, style_dim),
                                 nn.ReLU(),
                                 nn.Linear(style_dim, adain_num))
        print('start initialize')
        self.reset_params()
        print('end initialize')

    @staticmethod
    def weight_init(m, init_type='normal', gain=0.02):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                # m.weight.data *= scale
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward_style(self, style_image):
        style_feature = self.style_encoder(style_image)
        return style_feature

    def forward(self, input_feature, seg, style_image):
        style_feature_final = []
        for mm in range(len(style_image)):
            style_feature = self.style_encoder(style_image[mm])
            style_feature = style_feature.view(style_feature.shape[0], -1)
            style_feature_final.append(torch.mean(style_feature, dim=0).unsqueeze(dim=0))
        style_feature_final = torch.cat(style_feature_final, dim=0)

        adain_params = self.mlp(style_feature_final)
        self.assign_adain_params(adain_params, self.style_transfer)
        input_feature = self.style_transfer(input_feature)

        x = self.G_middle_0(input_feature, seg)
        x = self.atrous1(x)
        x = self.atrous2(x)
        x = self.atrous3(x)
        x = self.atrous4(x)
        rgb_64 = nn.Tanh()(self.conv_img(x))

        h_target = 128
        w_target = 128
        x = F.interpolate(x, size=[h_target, w_target], mode='bilinear')
        x = self.G_middle_1(x, seg)
        x = self.atrous1_conv(x)
        x = self.atrous2_conv(x)
        x = self.atrous3_conv(x)
        x = self.atrous4_conv(x)
        rgb_128 = nn.Tanh()(self.conv_img2(x))

        h_target = 256
        w_target = 256
        x = F.interpolate(x, size=[h_target, w_target], mode='bilinear')
        x = self.up_0(x, seg)
        x = self.atrous1_conv2(x)
        x = self.atrous2_conv2(x)
        x = self.atrous3_conv2(x)
        x = self.atrous4_conv2(x)
        x = nn.Tanh()(self.conv_img3(x))
        return x, rgb_128, rgb_64
