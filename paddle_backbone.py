import math
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D

from ..shape_spec import ShapeSpec

__all__ = ['micronet']
#################################################
# init_weight for unit layer in paddle
# self._linear = paddle.nn.Linear(32, 64)
# constant_init(self._linear.weight,value=1.)
######################################################
def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)

def get_size(x):
    x_shape=x.shape
    return x_shape[0],x_shape[1],x_shape[2],x_shape[3]

def normal_init(param, **kwargs):
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):   
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


def kaiming_uniform(param, **kwargs):
    initializer = nn.initializer.KaimingUniform(**kwargs)
    initializer(param, param.block)

def init_params(layer_list):
    for m in layer_list:
        
        if isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)     
            if m.bias is not None:
                constant_init(m.bias,value=0.)
        elif isinstance(m, nn.BatchNorm2D):
            constant_init(m.weight,value=1.)
            constant_init(m.bias,value=0.)
        elif isinstance(m, nn.Linear):
            n = m.weight.shape[1]
            normal_init(m.weight,mean=0.0, std=0.01)   
            if m.bias is not None:
                constant_init(m.bias,value=0.)   
        v=m.sublayers()
        if len(v)!=0:init_params(v)     

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Layer):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def get_act_layer(inp, oup, mode='SE1', act_relu=True, act_max=2, act_bias=True, init_a=[1.0, 0.0], reduction=4, init_b=[0.0, 0.0], g=None, act='relu', expansion=True):
    layer = None
    if mode == 'SE1':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction), 
            nn.ReLU6() if act_relu else nn.Sequential()
        )
    elif mode == 'SE0':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction), 
        )
    elif mode == 'NA':
        layer = nn.ReLU6() if act_relu else nn.Sequential()
    elif mode == 'LeakyReLU':
        layer = nn.LeakyReLU() if act_relu else nn.Sequential()
    elif mode == 'RReLU':
        layer = nn.RReLU() if act_relu else nn.Sequential()
    elif mode == 'PReLU':
        layer = nn.PReLU() if act_relu else nn.Sequential()
    elif mode == 'DYShiftMax':
        layer = DYShiftMax(inp, oup, act_max=act_max, act_relu=act_relu, init_a=init_a, reduction=reduction, init_b=init_b, g=g, expansion=expansion)
    return layer

########################################################################
# dynamic activation layers (SE, DYShiftMax, etc)
########################################################################

class SELayer(nn.Layer):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.oup = oup
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        # determine squeeze
        squeeze = get_squeeze_channels(inp, reduction)
        # # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))


        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(),
                nn.Linear(squeeze, oup),
                h_sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, _, _ = get_size(x_in)
        y = self.avg_pool(x_in).reshape([b, c])
        y = self.fc(y).reshape([b, self.oup, 1, 1])
        return x_out * y

class DYShiftMax(nn.Layer):
    def __init__(self, inp, oup, reduction=4, act_max=1.0, act_relu=True, init_a=[0.0, 0.0], init_b=[0.0, 0.0], relu_before_pool=False, g=None, expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
                nn.ReLU() if relu_before_pool == True else nn.Sequential(),
                nn.AdaptiveAvgPool2D(1)
            )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init-a: {}, init-b: {}'.format(init_a, init_b))

        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g !=1  and expansion:
            self.g = inp // self.g
        # print('group shuffle: {}, divide group: {}'.format(self.g, expansion))
        self.gc = inp//self.g
        list_pre=[i for i in range(inp)]
        index=paddle.to_tensor(list_pre,dtype='float32')
        # index=index.unsqueeze(0)
        # index=index.unsqueeze(2)
        # index=index.unsqueeze(3)
        index=index.reshape([1,inp,1,1])
        index=index.reshape([1,self.g,self.gc,1,1])
        indexgs = paddle.split(index, [1, self.g-1], axis=1)
        indexgs = paddle.concat((indexgs[1], indexgs[0]), axis=1)
        indexs = paddle.split(indexgs, [1, self.gc-1], axis=2)
        indexs = paddle.concat((indexs[1], indexs[0]), axis=2)
        self.index = indexs.reshape([inp])
        self.index=paddle.to_tensor(self.index,dtype='int64')
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = get_size(x_in)
        y = self.avg_pool(x_in).reshape([b, c])
        y = self.fc(y).reshape([b, self.oup*self.exp, 1, 1])
        y = (y-0.5) * self.act_max

        n2, c2, h2, w2 = get_size(x_out)
        x2=paddle.index_select(x_out, self.index, axis=1)
        if self.exp == 4:
            a1, b1, a2, b2 = paddle.split(y, y.shape[1]//self.oup, axis=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = paddle.maximum(z1, z2)

        elif self.exp == 2:
            a1, b1 = paddle.split(y, y.shape[1]//self.oup, axis=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out

def get_squeeze_channels(inp, reduction):
    if reduction == 4:
        squeeze = inp // reduction
    else:
        squeeze = _make_divisible(inp // reduction, 4)
    return squeeze

##########################################
#microcfg
#############################################

msnx_dy6_exp4_4M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4,y1,y2,y3,r
        [2, 1,   8, 3, 2, 2,  0,  4,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 4,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 4,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 4,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 384, 3, 1, 4, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]
msnx_dy6_exp6_6M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1,   8, 3, 2, 2,  0,  6,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  16, 3, 2, 2,  0,  8,  16,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 16,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 6,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 576, 3, 1, 6, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]
msnx_dy9_exp6_12M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 0, 1, 1], #8->16(0, 0)->32  ->12(4,3)->12
        [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #12->24(0,0)->48  ->16(8, 2)->16
        [1, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 2, 2, 1, 1], #16->16(0, 0)->64  ->24(8,3)->24
        [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 2, 2, 1, 1], #24->24(2, 12)->144  ->32(16,2)->32
        [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 2, 2, 1, 2], #32->32(2,16)->192 ->32(16,2)->32
        [1, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 2], #32->32(2,16)->192 ->64(12,4)->64
        [2, 1,  96, 5, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(4,12)->384 ->96(16,5)->96
        [1, 1, 128, 3, 1, 6, 12, 12, 128,  8,  8, 2, 2, 1, 2], #96->96(5,16)->576->128(16,8)->128
        [1, 1, 768, 3, 1, 6, 16, 16,   0,  0,  0, 2, 2, 1, 2], #128->128(4,32)->768
        ]
msnx_dy12_exp6_20M_020_cfgs = [
    #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 0, 2, 0, 1], #12->24(0, 0)->48  ->16(8,2)->16
    [2, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 0, 2, 0, 1], #16->32(0, 0)->64  ->24(8,3)->24
    [1, 1,  24, 3, 2, 2,  0, 24,  24,  4,  4, 0, 2, 0, 1], #24->48(0, 0)->96  ->24(8,3)->24
    [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 0, 2, 0, 1], #24->24(2,12)->144  ->32(16,2)->32
    [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 0, 2, 0, 2], #32->32(2,16)->192 ->32(16,2)->32
    [1, 1,  64, 5, 1, 6,  8,  8,  48,  8,  8, 0, 2, 0, 2], #32->32(2,16)->192 ->48(12,4)->48
    [1, 1,  80, 5, 1, 6,  8,  8,  80,  8,  8, 0, 2, 0, 2], #48->48(3,16)->288 ->80(16,5)->80
    [1, 1,  80, 5, 1, 6, 10, 10,  80,  8,  8, 0, 2, 0, 2], #80->80(4,20)->480->80(20,4)->80
    [2, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2], #80->80(4,20)->480->128(16,8)->128
    [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2], #120->128(4,32)->720->128(32,4)->120
    [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2], #120->128(4,32)->720->160(32,5)->144
    [1, 1, 864, 3, 1, 6, 12, 12,   0,  0,  0, 0, 2, 0, 2], #144->144(5,32)->864
]

def get_micronet_config(mode):
    return eval(mode+'_cfgs')

#############################################
#micro_cur_cfg
#############################################
from yacs.config import CfgNode as CN
def get_cfg():
  cfg=CN()
  cfg.MODEL=CN()
  cfg.MODEL.MICRONETS=CN()
  cfg.MODEL.ACTIVATION=CN()
  cfg.MODEL.MICRONETS.BLOCK="DYMicroBlock" 
  cfg.MODEL.MICRONETS.NET_CONFIG="msnx_dy12_exp6_20M_020" 
  cfg.MODEL.MICRONETS.STEM_CH=12 
  cfg.MODEL.MICRONETS.STEM_GROUPS=4,3
  cfg.MODEL.MICRONETS.STEM_DILATION=1 
  cfg.MODEL.MICRONETS.STEM_MODE="spatialsepsf" 
  cfg.MODEL.MICRONETS.OUT_CH=1024 
  cfg.MODEL.MICRONETS.DEPTHSEP=True 
  cfg.MODEL.MICRONETS.POINTWISE="group" 
  cfg.MODEL.MICRONETS.DROPOUT=0.1 
  cfg.MODEL.ACTIVATION.MODULE="DYShiftMax" 
  cfg.MODEL.ACTIVATION.ACT_MAX=2.0 
  cfg.MODEL.ACTIVATION.LINEARSE_BIAS=False 
  cfg.MODEL.ACTIVATION.INIT_A_BLOCK3=1.0,0.0 
  cfg.MODEL.ACTIVATION.INIT_A=1.0,0.5 
  cfg.MODEL.ACTIVATION.INIT_B=0.0,0.5 
  cfg.MODEL.ACTIVATION.REDUCTION=8
  cfg.MODEL.MICRONETS.SHUFFLE=True 
  return cfg

#############################################
#main_micronet
#############################################

def conv_3x3_bn(inp, oup, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 3, stride, 1, bias=False, dilation=dilation),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )

def gcd(a, b):
    a, b = (a, b) if a >= b else (b, a)
    while b:
        a, b = b, a%b
    return a

#####################################################################3
# part 2: Layers
#####################################################################3

class MaxGroupPooling(nn.Layer):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = get_size(x)

        # reshape
        y = x.reshape([b, c // self.channel_per_group, -1, h, w])
        out, _ = paddle.max(y, axis=2)
        return out

class SwishLinear(nn.Layer):
    def __init__(self, inp, oup):
        super(SwishLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1D(oup),
            h_swish()
        )

    def forward(self, x):
        return self.linear(x)

class StemLayer(nn.Layer):
    def __init__(self, inp, oup, stride, dilation=1, mode='default', groups=(4,4)):
        super(StemLayer, self).__init__()

        self.exp = 1 if mode == 'default' else 2
        g1, g2 = groups 
        if mode == 'default':
            self.stem = nn.Sequential(
                nn.Conv2D(inp, oup*self.exp, 3, stride, 1, bias=False, dilation=dilation),
                nn.BatchNorm2D(oup*self.exp),
                nn.ReLU6() if self.exp == 1 else MaxGroupPooling(self.exp)
            )
        elif mode == 'spatialsepsf':
            self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1*g2==2*oup else nn.ReLU6()
            )
        else: 
            raise ValueError('Undefined stem layer')
           
    def forward(self, x):
        out = self.stem(x)    
        return out

class GroupConv(nn.Layer):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        # print ('inp: %d, oup:%d, g:%d' %(inp, oup, self.groups[0]))
        self.conv = nn.Sequential(
            nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False, groups=self.groups[0]),
            nn.BatchNorm2D(oup)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelShuffle(nn.Layer):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = get_size(x)

        channels_per_group = c // self.groups

        # reshape
        x = x.reshape([b, self.groups, channels_per_group, h, w])

        x = paddle.transpose(x, [0,2,1,3,4])
        out = x.reshape([b, -1, h, w])

        return out

class ChannelShuffle2(nn.Layer):
    def __init__(self, groups):
        super(ChannelShuffle2, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = get_size(x)

        channels_per_group = c // self.groups

        # reshape
        x = x.reshape([b, self.groups, channels_per_group, h, w])

        x = paddle.transpose(x, [0,2,1,3,4])
        out = x.reshape([b, -1, h, w])

        return out

######################################################################3
# part 3: new block
#####################################################################3

class SpatialSepConvSF(nn.Layer):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2D(inp, oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size//2, 0),
                bias_attr=False, groups=1
            ),
            nn.BatchNorm2D(oup1),
            nn.Conv2D(oup1, oup1*oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias_attr=False, groups=oup1
            ),
            nn.BatchNorm2D(oup1*oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthConv(nn.Layer):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(inp, oup, kernel_size, stride, kernel_size//2, bias=False, groups=inp),
            nn.BatchNorm2D(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthSpatialSepConv(nn.Layer):
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand

        hidden_dim = inp*exp1
        oup = inp*exp1*exp2
        
        self.conv = nn.Sequential(
            nn.Conv2D(inp, inp*exp1, 
                (kernel_size, 1), 
                (stride, 1), 
                (kernel_size//2, 0), 
                bias_attr=False, groups=inp
            ),
            nn.BatchNorm2D(inp*exp1),
            nn.Conv2D(hidden_dim, oup,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias_attr=False, groups=hidden_dim
            ),
            nn.BatchNorm2D(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def get_pointwise_conv(mode, inp, oup, hiddendim, groups):

    if mode == 'group':
        return GroupConv(inp, oup, groups)
    elif mode == '1x1':
        return nn.Sequential(
                    nn.Conv2D(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2D(oup)
                )
    else:
        return None
 
class DYMicroBlock(nn.Layer):
    def __init__(self, inp, oup, kernel_size=3, stride=1, ch_exp=(2, 2), ch_per_group=4, groups_1x1=(1, 1), depthsep=True, shuffle=False, pointwise='fft', activation_cfg=None):
        super(DYMicroBlock, self).__init__()

        # print(activation_cfg.dy)

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg.dy
        act = activation_cfg.MODULE
        act_max = activation_cfg.ACT_MAX
        act_bias = activation_cfg.LINEARSE_BIAS
        act_reduction = activation_cfg.REDUCTION * activation_cfg.ratio
        init_a = activation_cfg.INIT_A
        init_b = activation_cfg.INIT_B
        init_ab3 = activation_cfg.INIT_A_BLOCK3

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle2(hidden_dim2//2) if shuffle and y2 !=0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and oup%2 == 0  and y3!=0 else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = gs1,
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),

            )

        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y1 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride) if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = True
                ) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle2(hidden_dim2//4) if shuffle and y1!=0 and y2 !=0 else nn.Sequential() if y1==0 and y2==0 else ChannelShuffle2(hidden_dim2//2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)), #FFTConv
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and y3!=0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = out + identity

        return out




@register
@serializable
class micronet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 ch_in=3,
                 variant='b',
                 input_size=224,
                 ch_out=[16,24,80,864],
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0,1,2,3],
                 dcn_v2_stages=[-1],
                 num_stages=4,
                 std_senet=False):
        super(micronet, self).__init__()   
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]  
        else : self.return_idx = return_idx   
        self.ch_in = ch_in
        self._out_channels = ch_out
        self._out_strides = [1,2,4,8]

        cfg=get_cfg()
        mode = cfg.MODEL.MICRONETS.NET_CONFIG
        self.cfgs = get_micronet_config(mode)

        block = eval(cfg.MODEL.MICRONETS.BLOCK)
        stem_mode = cfg.MODEL.MICRONETS.STEM_MODE
        stem_ch = cfg.MODEL.MICRONETS.STEM_CH
        stem_dilation = cfg.MODEL.MICRONETS.STEM_DILATION
        stem_groups = cfg.MODEL.MICRONETS.STEM_GROUPS
        out_ch = cfg.MODEL.MICRONETS.OUT_CH
        depthsep = cfg.MODEL.MICRONETS.DEPTHSEP
        shuffle = cfg.MODEL.MICRONETS.SHUFFLE
        pointwise = cfg.MODEL.MICRONETS.POINTWISE
        dropout_rate = cfg.MODEL.MICRONETS.DROPOUT

        act_max = cfg.MODEL.ACTIVATION.ACT_MAX
        act_bias = cfg.MODEL.ACTIVATION.LINEARSE_BIAS
        activation_cfg= cfg.MODEL.ACTIVATION

        # building first layer
        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [StemLayer(
                    3, input_channel,
                    stride=2, 
                    dilation=stem_dilation, 
                    mode=stem_mode,
                    groups=stem_groups
                )]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg.dy = [y1, y2, y3]
            activation_cfg.ratio = r

            output_channel = c
            layers.append(block(input_channel, output_channel,
                kernel_size=ks, 
                stride=s, 
                ch_exp=t1, 
                ch_per_group=gs1, 
                groups_1x1=gs2,
                depthsep = depthsep,
                shuffle = shuffle,
                pointwise = pointwise,
                activation_cfg=activation_cfg,
            ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 
                    kernel_size=ks, 
                    stride=1, 
                    ch_exp=t1, 
                    ch_per_group=gs1, 
                    groups_1x1=gs2,
                    depthsep = depthsep,
                    shuffle = shuffle,
                    pointwise = pointwise,
                    activation_cfg=activation_cfg,
                ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.layers=layers

        self.avgpool = nn.Sequential(
            nn.ReLU6(),
            nn.AdaptiveAvgPool2D((1, 1)),
            h_swish()
        ) 

        # building last several layers
        output_channel = out_ch
         
        # self.classifier = nn.Sequential(
        #     SwishLinear(input_channel, output_channel),
        #     nn.Dropout(dropout_rate),
        #     SwishLinear(output_channel, num_classes)
        # )
        # self._initialize_weights()

    def forward(self, inputs):
        x = inputs['image']
        outs=[]
        for i,v in enumerate(self.layers):
            x=v(x)
            if i==1 or i==3 or i==8 or i==12:outs.append(x)
        x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.classifier(x)
        return outs

    def _initialize_weights(self):
        init_params(self.sublayers)   

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]










