import paddle
import paddle.nn as nn
from .utils import  DropPath

class Aff(nn.Layer):
    def __init__(self, dim):
        super(Aff, self).__init__()
        # learnable
        self.alpha = paddle.base.framework.EagerParamBase.from_tensor(tensor
                                                                      =paddle.ones(shape=[1, 1, dim]))
        self.beta = paddle.base.framework.EagerParamBase.from_tensor(tensor
                                                                     =paddle.zeros(shape=[1, 1, dim]))
    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class Aff_channel(nn.Layer):
    def __init__(self, dim, channel_first=True):
        super(Aff_channel, self).__init__()
        tensor_alpha = paddle.ones(shape=[1, 1, dim])
        tensor_beta = paddle.zeros(shape=[1, 1, dim])
        tensor_color = paddle.eye(num_rows=dim)

        # 使用 create_parameter 创建参数
        self.alpha = self.create_parameter(
            shape=tensor_alpha.shape,
            dtype=tensor_alpha.dtype,
            default_initializer=paddle.nn.initializer.Assign(tensor_alpha)
        )

        self.beta = self.create_parameter(
            shape=tensor_beta.shape,
            dtype=tensor_beta.dtype,
            default_initializer=paddle.nn.initializer.Assign(tensor_beta)
        )

        self.color = self.create_parameter(
            shape=tensor_color.shape,
            dtype=tensor_color.dtype,
            default_initializer=paddle.nn.initializer.Assign(tensor_color)
        )
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x1 = paddle.tensordot(x, self.color, axes=[[x.ndim - 1], [1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x2 = x * self.alpha + self.beta
            x1 = paddle.tensordot(x2, self.color, axes=[[x.ndim - 1], [1]])
        return x1

class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(CMlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class CBlock_ln(nn.Layer):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-4):
        super(CBlock_ln, self).__init__()
        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = Aff_channel(dim) 
        self.conv1 = nn.Conv2D(dim, dim, 1)
        self.conv2 = nn.Conv2D(dim, dim, 1)
        self.attn = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Aff_channel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gamma_1 = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=paddle.nn.initializer.Constant(value=init_values))
        self.gamma_2 = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=paddle.nn.initializer.Constant(value=init_values))
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        norm_x = x.flatten(start_axis=2).transpose([0, 2, 1])
        norm_x = self.norm1(norm_x)
        norm_x = norm_x.reshape([B, H, W, C]).transpose([0, 3, 1, 2])

        x = x + self.drop_path(self.gamma_1 * self.conv2(self.attn(self.conv1(norm_x))))
        norm_x = x.flatten(start_axis=2).transpose([0, 2, 1])
        norm_x = self.norm2(norm_x)
        norm_x = norm_x.reshape([B, H, W, C]).transpose([0, 3, 1, 2])

        x = x + self.drop_path(self.gamma_2 * self.mlp(norm_x))
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1])
    return x

class WindowAttention(nn.Layer):
    r"""
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape([B_, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # make paddle script happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Layer):
    def __init__(self, dim, num_heads=2, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = Aff_channel(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Aff_channel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(start_axis=2).transpose([0, 2, 1])

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = shifted_x
        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x