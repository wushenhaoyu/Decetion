import os
import paddle
import argparse
from tqdm import tqdm
import cv2
from util import slice_gauss, map_range, cv2torch, random_tone_map, DirectoryDataset, str2bool
from model import ExpandNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help=
        'Batch size.')
    parser.add_argument('--checkpoint_freq', type=int, default=200, help=
        'Checkpoint model every x epochs.')
    parser.add_argument('-d', '--data_root_path', default='hdr_data', help=
        'Path to hdr data.')
    parser.add_argument('-s', '--save_path', default='checkpoints', help=
        'Path for checkpointing.')
    parser.add_argument('--num_workers', type=int, default=4, help=
        'Number of data loading workers.')
    parser.add_argument('--loss_freq', type=int, default=20, help=
        'Report (average) loss every x iterations.')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help=
        'Use GPU for training.')
    return parser.parse_args()


class ExpandNetLoss(paddle.nn.Layer):

    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = paddle.nn.CosineSimilarity(axis=1, eps=1e-20)
        self.l1_loss = paddle.nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


def transform(hdr):
    hdr = slice_gauss(hdr, crop_size=(384, 384), precision=(0.1, 1))
    hdr = cv2.resize(hdr, (256, 256))
    hdr = map_range(hdr)
    ldr = random_tone_map(hdr)
    return cv2torch(ldr), cv2torch(hdr)


def train(opt):
    model = ExpandNet()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=7e-05, weight_decay=0.0)
    loss = ExpandNetLoss()
    dataset = DirectoryDataset(data_root_path=opt.data_root_path,
        preprocess=transform)
    loader = paddle.io.DataLoader(dataset=dataset, batch_size=opt.
        batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    if opt.use_gpu:
        model.cuda(blocking=True)
        # False = True
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print(
            'WARNING: save_path already exists. Checkpoints may be overwritten'
            )
    avg_loss = 0
    for epoch in tqdm(range(1, 10001), desc='Training'):
        for i, (ldr_in, hdr_target) in enumerate(tqdm(loader, desc=
            f'Epoch {epoch}')):
            if opt.use_gpu:
                ldr_in = ldr_in.cuda(blocking=True)
                hdr_target = hdr_target.cuda(blocking=True)
            hdr_prediction = model(ldr_in)
            total_loss = loss(hdr_prediction, hdr_target)
            optimizer.clear_gradients(set_to_zero=False)
            total_loss.backward()
            optimizer.step()
            avg_loss += total_loss.item()
            if (i + 1) % opt.loss_freq == 0:
                rep = (
                    f'Epoch: {epoch:>5d}, Iter: {i + 1:>6d}, Loss: {avg_loss / opt.loss_freq:>6.2e}'
                    )
                tqdm.write(rep)
                avg_loss = 0
        if epoch % opt.checkpoint_freq == 0:
            paddle.save(obj=model.state_dict(), path=os.path.join(opt.
                save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    opt = parse_args()
    train(opt)
