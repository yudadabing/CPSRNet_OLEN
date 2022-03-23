import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn.modules import module

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


# gated Spatial conv2d from GSCNN
class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, gate_channels, out_channels, kernel_size=1, bn=False, stride=1, padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, 'zeros')

        module_body=[]

        if bn:
            module_body.append(nn.BatchNorm2d(in_channels + gate_channels))

        module_body.append(nn.Conv2d(in_channels + gate_channels, in_channels + gate_channels, 1))
        module_body.append(nn.ReLU()),
        module_body.append(nn.Conv2d(in_channels + gate_channels, 1, 1))

        if bn:
            module_body.append(nn.BatchNorm2d(1))

        module_body.append(nn.Sigmoid())

        self._gate_conv = nn.Sequential(*module_body)

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

##add noise

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 


def Conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()

        modules_body = [Conv(n_feat, n_feat, kernel_size, bias=bias), act, Conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.noise=GaussianNoise()
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x

        # return self.noise(res.mul(0.2) + x)   #添加噪声在生成器中
        return res


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        # self.noise=GaussianNoise()
    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        
        res += x
        # res=self.noise(res )  
        
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.noise=GaussianNoise()
    def forward(self, x):
        res = self.body(x)
        res += x
        # res=self.noise(res )     
        return res


class FuseModules(nn.Module):  ##fusion-net
    def __init__(self, conv=Conv, n_feat=64, kernel_size=3, act=nn.ReLU(True), bias=False):
        super(FuseModules, self).__init__()
        modules_body = [
                        conv(n_feat, n_feat, kernel_size, bias=bias), act,
                        conv(n_feat, n_feat, kernel_size, bias=bias), act,
                        conv(n_feat, n_feat, kernel_size, bias=bias), act]
        self.body = nn.Sequential(*modules_body)

    def forward(self, s_features, d_features):
        res = self.body(d_features - s_features)
        res = res + s_features

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class PRNet(nn.Module):
    def __init__(self, rgb_c=3, polarity_c=3, out_c=3, upscale=8, n_feats=64,
                 kernel_size=3, reduction=4, bias=True, res_scale=1, n_resgroups=10, n_resblocks=20,
                 act=nn.ReLU(True), conv=Conv):
        super(PRNet, self).__init__()

        self.shallow_feat1 = nn.Sequential(conv(rgb_c, n_feats, kernel_size, bias=bias),
                                           CAB(n_feats, kernel_size, reduction, bias=bias, act=act))

        self.shallow_feat2 = nn.Sequential(conv(polarity_c, n_feats, kernel_size, bias=bias),
                                           CAB(n_feats, kernel_size, reduction, bias=bias, act=act))

        self.polarity_feats = nn.Sequential(ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=10))

        self.gsc = GatedSpatialConv2d(in_channels=n_feats, gate_channels=n_feats, out_channels=n_feats)

        self.module_fusion = FuseModules(conv=Conv, n_feat=n_feats, kernel_size=kernel_size, act=act, bias=bias)

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, upscale, n_feats, act=False),
            conv(n_feats, out_c, kernel_size)]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x_img, p_img):
        # -------------- polarity ---------------------
        p_shallow_f = self.shallow_feat2(p_img)
        p_f = self.polarity_feats(p_shallow_f)

        # -------------- rgb -------------------
        x_shallow_f = self.shallow_feat1(x_img)
        
        # Cross-Branch Activation Module(CBAM)        
        x_gated_f = self.gsc(input_features=x_shallow_f, gating_features=p_f)
        
        #Related-Supervised Residual Fusion Module(RSRFM)
        fused = self.module_fusion(x_gated_f, p_f )
        
        #deep feature extraction(residual in residua)
        x = self.body(fused)
        x = x + x_shallow_f 

        #reconstruction part
        out = self.tail(x)

        return [out, fused,x_shallow_f]##two input,fused feature maps and shallow feature of S0.

    



