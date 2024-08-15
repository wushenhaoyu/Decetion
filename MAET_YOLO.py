import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import random
import scipy.stats as stats


def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    # print('shot noise and read noise:', log_shot_noise, log_read_noise)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

# hook gradient
grads = {}
def save_grad(name):
    def hook_fn(grad):
        #print(grad)
        grads[name] = grad
        return grad
    return hook_fn

@paddle.nn.Layer
class MAET_YOLO(nn.Layer):
    def __init__(self, backbone, neck, bbox_head, aet=None, ort_cfg=None, degration_cfg=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(MAET_YOLO, self).__init__()

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.aet = build_shared_head(aet)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.degration_cfg = degration_cfg
        self.use_ori = ort_cfg['use_ort']
        self.init_weights(pretrained=pretrained)
        self.loss_ort = nn.CosineSimilarity(axis=1)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector."""
        super(MAET_YOLO, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if hasattr(self, 'neck'):
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def apply_ccm(self, image, ccm):
        '''Apply CCM matrix'''
        shape = image.shape
        image = image.flatten(start_axis=1)
        image = paddle.tensordot(image, ccm, axes=[[-1], [-1]])
        return image.reshape(shape)

    def weight_L1_loss(self, pred, gt, weight=5):
        '''Calculate weighted L1 loss'''
        loss = weight * paddle.mean((pred[:, 0:1] - gt[:, 0:1]) ** 2) + paddle.mean((pred[:, 1:] - gt[:, 1:]) ** 2)
        return loss

    def Low_Illumination_Degrading(self, img, img_meta, safe_invert=False):
        '''Generate low light degraded images and parameters'''
        device = paddle.get_device()
        config = self.degration_cfg
        xyz2cams = [...]  # Same as above
        rgb2xyz = [...]

        img1 = img.transpose((1, 2, 0))  # (C, H, W) to (H, W, C)
        img1 = 0.5 - paddle.sin(paddle.asin(1.0 - 2.0 * img1) / 3.0)
        epsilon = paddle.to_tensor([1e-8], dtype=paddle.float32)
        gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
        img2 = paddle.maximum(img1, epsilon) ** gamma
        xyz2cam = random.choice(xyz2cams)
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = paddle.to_tensor(rgb2cam / np.sum(rgb2cam, axis=-1), dtype=paddle.float32)
        img3 = self.apply_ccm(img2, rgb2cam)
        rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
        red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
        blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])
        gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
        gains1 = paddle.to_tensor(gains1[np.newaxis, np.newaxis, :], dtype=paddle.float32)
        if safe_invert:
            img3_gray = paddle.mean(img3, axis=-1, keepdim=True)
            inflection = 0.9
            zero = paddle.zeros_like(img3_gray)
            mask = (paddle.maximum(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
            safe_gains = paddle.maximum(mask + (1.0 - mask) * gains1, gains1)
            img4 = paddle.clip(img3 * safe_gains, min=0.0, max=1.0)
        else:
            img4 = img3 * gains1

        lower, upper = config['darkness_range'][0], config['darkness_range'][1]
        mu, sigma = 0.1, 0.08
        darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        darkness = darkness.rvs()
        img5 = img4 * darkness
        shot_noise, read_noise = random_noise_levels()
        var = img5 * shot_noise + read_noise
        var = paddle.maximum(var, epsilon)
        noise = paddle.normal(mean=0, std=paddle.sqrt(var))
        img6 = img5 + noise

        bits = random.choice(config['quantisation'])
        quan_noise = paddle.to_tensor(img6.shape, dtype=paddle.float32).uniform_(-1 / (255 * bits), 1 / (255 * bits))
        img7 = img6 + quan_noise
        gains2 = np.stack([red_gain, 1.0, blue_gain])
        gains2 = paddle.to_tensor(gains2[np.newaxis, np.newaxis, :], dtype=paddle.float32)
        img8 = img7 * gains2
        cam2rgb = paddle.linalg.inv(rgb2cam)
        img9 = self.apply_ccm(img8, cam2rgb)
        img10 = paddle.maximum(img9, epsilon) ** (1 / gamma)

        img_low = img10.transpose((2, 0, 1))  # (H, W, C) to (C, H, W)
        para_gt = paddle.to_tensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain], dtype=paddle.float32)
        return img_low, para_gt

    def extract_feat_aet(self, img, img_dark):
        """Extract features and AET predictions"""
        x_light = self.backbone(img)
        x_dark = self.backbone(img_dark)
        feat = x_light[2]
        feat1 = x_dark[2]
        para_pred = self.aet(feat, feat1)
        x_light[2].register_hook(save_grad('light_grad'))
        x_dark[2].register_hook(save_grad('dark_grad'))
        if hasattr(self, 'neck'):
            x_dark = self.neck(x_dark)
        return x_dark, para_pred

    def extract_feat(self, img_dark):
        """Extract features from dark images"""
        x_dark = self.backbone(img_dark)
        if hasattr(self, 'neck'):
            x_dark = self.neck(x_dark)
        return x_dark

    def forward_dummy(self, img, img_dark):
        """Dummy forward pass for computing network flops"""
        x, _ = self.extract_feat(img_dark)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        """Forward pass for training"""
        batch_size = img.shape[0]
        img_dark = paddle.empty([batch_size, img.shape[1], img.shape[2], img.shape[3]], dtype=paddle.float32)
        para_gt = paddle.empty([batch_size, 4], dtype=paddle.float32)
        for i in range(batch_size):
            img_dark[i], para_gt[i] = self.Low_Illumination_Degrading(img[i], img_metas[i])
        x_dark, para_pred = self.extract_feat_aet(img, img_dark)
        losses = self.bbox_head.forward_train(x_dark, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        aet_loss = 10 * self.weight_L1_loss(para_pred, para_gt)
        losses['loss_aet'] = aet_loss
        if not grads:
            ort_loss = paddle.to_tensor([0.0], dtype=paddle.float32)
        if grads:
            ort_loss = 5 * paddle.mean(paddle.abs(self.loss_ort(grads['light_grad'].flatten(start_axis=1), grads['dark_grad'].flatten(start_axis=1)))) + \
                       0.5 * paddle.mean(1 - paddle.abs(self.loss_ort(grads['light_grad'].flatten(start_axis=1), grads['dark_grad'].flatten(start_axis=1))))
        losses['loss_ort'] = ort_loss
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Simple test function"""
        x_dark = self.extract_feat(img)
        bbox_inputs = x_dark
        bbox_outputs = self.bbox_head.simple_test(bbox_inputs, img_meta, rescale)
        return bbox_outputs
    def aug_test(self, imgs, img_metas, rescale=False):

        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'
        print('11111',imgs)
        # img_dark = torch.stack([self.Low_Illumination_Degrading(img[i], img_metas[i])[0]  for i in range(img.shape[0])], dim=0)
        # feate, _ = self.extract_feat(img, img_dark)
        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, 