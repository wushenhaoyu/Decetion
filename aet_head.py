import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, load_checkpoint

from mmdet.models.builder import SHARED_HEADS, build_loss
from mmdet.utils import get_root_logger

@SHARED_HEADS.register_module()
class AETHead(nn.Layer):
    """The decoder of AET branches, input the feat of original images and 
    feat of transformed images, passed by global pool and return the transformed
    results."""
    def __init__(self,
                 indim=2048, 
                 num_classes=4):
        super(AETHead, self).__init__()
        self.indim = indim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(indim, int(indim / 2))
        self.fc2 = nn.Linear(int(indim / 2), num_classes)
        
        self._init_weights()

    def global_pool(self, feat):
        num_channels = feat.shape[1]
        return F.adaptive_avg_pool2d(feat, 1).reshape([-1, num_channels])
    
    def forward(self, feat1, feat2):
        feat1 = self.global_pool(feat1)
        feat2 = self.global_pool(feat2)
        x = paddle.concat([feat1, feat2], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in LinearBlock
        pass
    

        


