import os
import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.vision.models import vgg16
from paddle.io import DataLoader
from utils.utils import LossNetwork, validation
from dataloader.data_loader import LowlightLoader
from tqdm import tqdm
from model import IAT

class IATTrainer:
    def __init__(self):
        # 配置直接写入类中
        self.config = {
            'gpu_id': 0,
            'img_path': os.path.join(os.getcwd(), "data", "train", "Low"),
            'img_val_path': os.path.join(os.getcwd(), "data", "eval", "Low"),
            'normalize': True,
            'model_type': 's',
            'batch_size': 8,
            'lr': 1e-4,
            'weight_decay': 0.0001,
            'pretrain_dir': None,
            'num_epochs': 400,
            'display_iter': 10,
            'snapshots_folder': "workdirs/snapshots_folder_lol_v1_patch"
        }

        self._setup_environment()
        self.model = self._initialize_model()
        self.vgg_model = self._initialize_vgg()
        self.loss_network = LossNetwork(self.vgg_model)
        self.optimizer, self.scheduler = self._initialize_optimizer()
        self.L1_smooth_loss = nn.functional.smooth_l1_loss

        # Datasets and dataloaders
        self.train_loader, self.val_loader = self._setup_data()

        self.ssim_high = 0
        self.psnr_high = 0

    def _setup_environment(self):
        # 设置Paddle设备
        paddle.set_device('gpu:' + str(self.config['gpu_id']))
        # 确保输出目录存在
        if not os.path.exists(self.config['snapshots_folder']):
            os.makedirs(self.config['snapshots_folder'])

    def _initialize_model(self):
        # 模型初始化
        model = IAT()
        model.load_dict(paddle.load('transform_paddle.pdparams'))
        model = model.to(paddle.CUDAPlace(0))

        # 加载预训练模型
        if self.config['pretrain_dir'] is not None:
            model.set_state_dict(paddle.load(self.config['pretrain_dir']))
        return model

    def _initialize_vgg(self):
        # VGG特征网络
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model.eval()
        vgg_model = vgg_model.to(paddle.CUDAPlace(0))

        # 冻结VGG特征网络的参数
        for param in vgg_model.parameters():
            param.stop_gradient = True
        return vgg_model

    def _initialize_optimizer(self):
        # 优化器与调度器
        optimizer = Adam(parameters=self.model.parameters(), learning_rate=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = CosineAnnealingDecay(learning_rate=self.config['lr'], T_max=self.config['num_epochs'])
        return optimizer, scheduler

    def _setup_data(self):
        # 加载训练和验证数据集
        train_dataset = LowlightLoader(images_path=self.config['img_path'], normalize=self.config['normalize'])
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=8)

        val_dataset = LowlightLoader(images_path=self.config['img_val_path'], mode='test', normalize=self.config['normalize'])
        val_loader = paddle.io.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
        return train_loader, val_loader

    def train(self):
        self.model.train()
        print('######## Start IAT Training #########')

        for epoch in range(self.config['num_epochs']):
            print('Epoch:', epoch)
            for iteration, imgs in enumerate(self.train_loader):
                low_img, high_img = imgs[0].cuda(), imgs[1].cuda()

                self.optimizer.clear_gradients()
                mul, add, enhance_img = self.model(low_img)

                loss = self.L1_smooth_loss(enhance_img, high_img) + 0.04 * self.loss_network(enhance_img, high_img)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if (iteration + 1) % self.config['display_iter'] == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())

            self.validate_and_save(epoch)

    def validate_and_save(self, epoch):
        # 验证模型
        self.model.eval()
        PSNR_mean, SSIM_mean = validation(self.model, self.val_loader)

        # 保存日志
        with open(self.config['snapshots_folder'] + '/log.txt', 'a+') as f:
            f.write(f'Epoch {epoch}: SSIM: {SSIM_mean}, PSNR: {PSNR_mean}\n')

        # 保存最高SSIM的模型
        if SSIM_mean > self.ssim_high:
            self.ssim_high = SSIM_mean
            print('Highest SSIM so far:', self.ssim_high)
            paddle.save(self.model.state_dict(), os.path.join(self.config['snapshots_folder'], "best_Epoch.pdparams"))


# 直接在外部调用类
trainer = IATTrainer()
trainer.train()
