import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle.regularizer import L2Decay

class ConvModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=True,
                 norm_type='BN',
                 act_type='LeakyReLU'):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              weight_attr=paddle.ParamAttr(regularizer=L2Decay(0.)),
                              bias_attr=bias_attr)
        
        if norm_type == 'BN':
            self.norm = nn.BatchNorm2D(out_channels)
        else:
            self.norm = None
        
        if act_type == 'LeakyReLU':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x
    
class ResBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 norm_type='BN',
                 act_type='LeakyReLU'):
        super(ResBlock, self).__init__()
        self.conv1 = ConvModule(in_channels, in_channels // 2, 1, norm_type=norm_type, act_type=act_type)
        self.conv2 = ConvModule(in_channels // 2, in_channels, 3, padding=1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        return x


class Darknet(nn.Layer):
    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)))
    }

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 norm_type='BN',
                 act_type='LeakyReLU',
                 norm_eval=True):
        super(Darknet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')
        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.layers, self.channels = self.arch_settings[depth]

        self.conv1 = ConvModule(3, 32, 3, padding=1, norm_type=norm_type, act_type=act_type)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_sublayer(
                layer_name,
                self.make_conv_res_block(in_c, out_c, n_layers, norm_type=norm_type, act_type=act_type))
            self.cr_blocks.append(layer_name)

        self.norm_eval = norm_eval

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # Load pretrained weights
            pass  # Implement loading logic here
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    nn.initializer.KaimingNormal()(m.weight)
                elif isinstance(m, nn.BatchNorm2D):
                    nn.initializer.Constant(value=1)(m.weight)
                    nn.initializer.Constant(value=0)(m.bias)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.stop_gradient = True

    def train(self, mode=True):
        super(Darknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m.eval()

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            norm_type='BN',
                            act_type='LeakyReLU'):
        model = nn.Sequential()
        model.add_sublayer(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, norm_type=norm_type, act_type=act_type))
        for idx in range(res_repeat):
            model.add_sublayer('res{}'.format(idx),
                               ResBlock(out_channels, norm_type=norm_type, act_type=act_type))
        return model
