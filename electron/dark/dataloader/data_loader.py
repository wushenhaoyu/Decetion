import os
import os.path as osp
import random
import numpy as np
from PIL import Image
from enum import Enum
import glob
import paddle
import paddle.vision.transforms as T
from paddle.vision.transforms import Compose, ToTensor, Normalize

# 设置随机种子
random.seed(1143)
def convert_dtype(img):
    return img.astype('float32')
def populate_train_list(images_path, mode='train'):
    image_list_lowlight = glob.glob( os.path.join(images_path, '*.png'))
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)
    return train_list

class LowlightLoader(paddle.io.Dataset):
    def __init__(self, images_path, mode='train', normalize=True):
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
        print("Total examples:", len(self.train_list))

    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high

    def get_params(self, low):
        self.w, self.h = low.size
        self.crop_height = random.randint(self.h // 2, self.h)
        self.crop_width = random.randint(self.w // 2, self.w)
        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params(low)
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        
        if self.mode == 'train':
            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('Low', 'Normal').replace('low', 'normal'))
            
            data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)
            
            data_lowlight = data_lowlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_highlight = data_highlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

            if self.normalize:
                transform_input = Compose([
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    convert_dtype
                ])

                transform_gt = Compose([
                    ToTensor(),
                    convert_dtype
                ])
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight = paddle.to_tensor(data_lowlight).astype(paddle.float32)
                data_highlight = paddle.to_tensor(data_highlight).astype(paddle.float32)
                return data_lowlight.transpose((2, 0, 1)), data_highlight.transpose((2, 0, 1))

        elif self.mode == 'test':
            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('Low', 'Normal').replace('low', 'normal'))
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)
            
            if self.normalize:
                transform_input = Compose([
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    convert_dtype
                ])

                transform_gt = Compose([
                    ToTensor(),
                    convert_dtype
                ])
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight = paddle.to_tensor(data_lowlight).astype(paddle.float32)
                data_highlight = paddle.to_tensor(data_highlight).astype(paddle.float32)
                return data_lowlight.transpose((2, 0, 1)), data_highlight.transpose((2, 0, 1))

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    images_path = '/data/unagi0/cui_data/light_dataset/LOL_v2/train/Low/'

    train_dataset = LowlightLoader(images_path)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    for iteration, (lowlight_images, highlight_images) in enumerate(train_loader):
        print(iteration)
        print(lowlight_images.shape)
        print(highlight_images.shape)