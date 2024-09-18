import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def _norm_cdf(x):
        # 将 x 转换为 tensor 类型
        x = paddle.to_tensor(x, dtype='float32')
        return (1. + paddle.erf(x / math.sqrt(2.))) / 2.

    # Get the bounds for truncation in standard normal space
    low = _norm_cdf((a - mean) / std)
    high = _norm_cdf((b - mean) / std)

    # Fill tensor with uniform numbers from [Low, Normal]
    uniform = paddle.uniform(tensor.shape, min=low, max=high)

    # Use inverse CDF transform for truncated normal distribution
    tensor = paddle.clip(uniform, low, high)
    tensor = paddle.erfinv(2 * tensor - 1) * math.sqrt(2) * std + mean

    # Clip to ensure no values fall outside the truncation range
    tensor = paddle.clip(tensor, min=a, max=b)

    return tensor

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)
        keep_prob_tensor = paddle.to_tensor(keep_prob, dtype=x.dtype)
        output = x.divide(keep_prob_tensor) * random_tensor
        return output

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)