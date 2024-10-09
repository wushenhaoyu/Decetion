import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .utils import trunc_normal_, DropPath
from .block import Mlp


class query_Attention(nn.Layer):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(query_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = self.create_parameter(shape=[1, 10, dim], default_initializer=nn.initializer.Constant(value=1.0))
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # 计算 q, k 和 v
        k = self.k(x).reshape([B, N, self.num_heads, head_dim]).transpose([0, 2, 1, 3])
        v = self.v(x).reshape([B, N, self.num_heads, head_dim]).transpose([0, 2, 1, 3])

        q = self.q.expand([B, -1, -1]).reshape([B, -1, self.num_heads, head_dim]).transpose([0, 2, 1, 3])

        # 计算注意力得分
        attn = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # 应用注意力得分
        attn_v = paddle.matmul(attn, v)

        # 打印 attn_v 的形状以确认维度
        # print(attn_v.shape)

        # 假设 attn_v 的维度是 (B, N, C, D)，调整 perm
        attn_v_transposed = paddle.transpose(attn_v, perm=[0, 2, 1, 3])

        # 确认重塑的形状与实际维度匹配
        x = paddle.reshape(attn_v_transposed, shape=[B, 10, C])
        # x = paddle.reshape(paddle.transpose(paddle.matmul(attn, v), perm=[0, 2, 1]), shape=[B, 10, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class query_SABlock(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(query_SABlock, self).__init__()
        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class conv_embedding(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2D(out_channels // 2),
            nn.GELU(),
            nn.Conv2D(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2D(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class Global_pred(nn.Layer):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4, type='exp'):
        super(Global_pred, self).__init__()
        if type == 'exp':
            self.gamma_base = paddle.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=1.0)
            )
        else:
            self.gamma_base = paddle.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=1.0)
            )
        self.color_base = paddle.create_parameter(
            shape=[3, 3],  # 形状是 3x3
            dtype='float32',  # 数据类型
            default_initializer=paddle.nn.initializer.Assign(paddle.eye(3))  # 初始化为单位矩阵
        )
        # main blocks
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                paddle.assign(paddle.zeros_like(m.bias), m.bias)
        elif isinstance(m, nn.LayerNorm):
            paddle.assign(paddle.zeros_like(m.bias), m.bias)
            paddle.assign(paddle.ones_like(m.weight), m.weight)

    def forward(self, x):
        x = self.conv_large(x)
        x = self.generator(x)
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        color = self.color_linear(color).squeeze(-1).reshape([-1, 3, 3]) + self.color_base
        return gamma, color

if __name__ == "__main__":
    img = paddle.randn([8, 3, 400, 600])
    global_net = Global_pred()
    gamma, color = global_net(img)
    print(gamma.shape, color.shape)