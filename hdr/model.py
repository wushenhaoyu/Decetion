import paddle
import numpy as np


class ExpandNet(paddle.nn.Layer):

    def __init__(self):
        super(ExpandNet, self).__init__()

        def layer(nIn, nOut, k, s, p, d=1):
            return paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=nIn,
                out_channels=nOut, kernel_size=k, stride=s, padding=p,
                dilation=d), paddle.nn.SELU())
        self.nf = 64
        self.local_net = paddle.nn.Sequential(layer(3, 64, 3, 1, 1), layer(
            64, 128, 3, 1, 1))
        self.mid_net = paddle.nn.Sequential(layer(3, 64, 3, 1, 2, 2), layer
            (64, 64, 3, 1, 2, 2), layer(64, 64, 3, 1, 2, 2), paddle.nn.
            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1,
            padding=2, dilation=2))
        self.glob_net = paddle.nn.Sequential(layer(3, 64, 3, 2, 1), layer(
            64, 64, 3, 2, 1), layer(64, 64, 3, 2, 1), layer(64, 64, 3, 2, 1
            ), layer(64, 64, 3, 2, 1), layer(64, 64, 3, 2, 1), paddle.nn.
            Conv2D(in_channels=64, out_channels=64, kernel_size=4, stride=1,
            padding=0))
        self.end_net = paddle.nn.Sequential(layer(256, 64, 1, 1, 0), paddle
            .nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1,
            stride=1, padding=0), paddle.nn.Sigmoid())

    def forward(self, x):
        local = self.local_net(x)
        mid = self.mid_net(x)
        resized = paddle.nn.functional.interpolate(x=x, size=(256, 256),
            mode='bilinear', align_corners=False)
        b, c, h, w = tuple(local.shape)
        glob = self.glob_net(resized).expand(shape=[b, 64, h, w])
        fuse = paddle.concat(x=(local, mid, glob), axis=-3)
        return self.end_net(fuse)

    def predict(self, x, patch_size):
        with paddle.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(axis=0)
            if x.shape[-3] == 1:
                x = x.expand(shape=[1, 3, *tuple(x.shape)[-2:]])
            resized = paddle.nn.functional.interpolate(x=x, size=(256, 256),
                mode='bilinear', align_corners=False)
            glob = self.glob_net(resized)
            overlap = 20
            skip = int(overlap / 2)
            result = x.clone()
            x = paddle.nn.functional.pad(x=x, pad=(skip, skip, skip, skip),
                pad_from_left_axis=False)
            padded_height, padded_width = x.shape[-2], x.shape[-1]
            num_h = int(np.ceil(padded_height / (patch_size - overlap)))
            num_w = int(np.ceil(padded_width / (patch_size - overlap)))
            for h_index in range(num_h):
                for w_index in range(num_w):
                    h_start = h_index * (patch_size - overlap)
                    w_start = w_index * (patch_size - overlap)
                    h_end = min(h_start + patch_size, padded_height)
                    w_end = min(w_start + patch_size, padded_width)
                    x_slice = x[:, :, h_start:h_end, w_start:w_end]
                    loc = self.local_net(x_slice)
                    mid = self.mid_net(x_slice)
                    exp_glob = glob.expand(shape=[1, 64, h_end - h_start, 
                        w_end - w_start])
                    fuse = paddle.concat(x=(loc, mid, exp_glob), axis=1)
                    res = self.end_net(fuse).data
                    h_start_stitch = h_index * (patch_size - overlap)
                    w_start_stitch = w_index * (patch_size - overlap)
                    h_end_stitch = min(h_start + patch_size - overlap,
                        padded_height)
                    w_end_stitch = min(w_start + patch_size - overlap,
                        padded_width)
                    res_slice = res[:, :, skip:-skip, skip:-skip]
                    paddle.assign(res_slice, output=result[:, :,
                        h_start_stitch:h_end_stitch, w_start_stitch:
                        w_end_stitch])
                    del fuse, loc, mid, res
            return result[0]
