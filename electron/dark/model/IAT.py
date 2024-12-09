import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from .utils import trunc_normal_
from .block import CBlock_ln, SwinTransformerBlock
from .global_net import Global_pred

class Local_pred(nn.Layer):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2D(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2D(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2D(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)
        return mul, add

class Local_pred_S(nn.Layer):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2D(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)
        self.mul_end = nn.Sequential(nn.Conv2D(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2D(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.initializer.Constant(value=0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.initializer.Constant(value=0)(m.bias)
            nn.initializer.Constant(value=1.0)(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            m.weight.set_value(paddle.normal(mean=0, std=math.sqrt(2.0 / fan_out), shape=m.weight.shape))
            if m.bias is not None:
                m.bias.set_value(paddle.zeros_like(m.bias))

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)
        return mul, add

class IAT(nn.Layer):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(IAT, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.reshape([-1, 3])
        image = paddle.tensordot(image, ccm, axes=[[1], [1]])
        image = image.reshape(shape)
        return paddle.clip(image, min=1e-8, max=1.0)

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = (img_low * mul) + add

        if not self.with_global:
            return mul, add, img_high
        
        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.transpose([0, 2, 3, 1])  # (B,C,H,W) -- (B,H,W,C)
            img_high = paddle.stack([self.apply_color(img_high[i], color[i])**gamma[i] for i in range(b)], axis=0)
            img_high = img_high.transpose([0, 3, 1, 2])  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high

if __name__ == "__main__":
    import os
    img = paddle.rand([1, 3, 400, 600])
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)