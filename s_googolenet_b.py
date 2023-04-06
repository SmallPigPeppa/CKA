from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.models import resnet50
import torchvision.transforms as transforms
import torch.nn.functional as F

from cka import CKA_Minibatch_Grid
import numpy as np

import os
import torch

from torchvision.models import resnet50
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from torchvision.models import vgg16, googlenet


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.model = googlenet(pretrained=True)
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)

    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        small_imgs = F.interpolate(small_imgs, size=self.large_size, mode='bilinear')
        mid_imgs = F.interpolate(mid_imgs, size=self.large_size, mode='bilinear')

        y1 = self.model(small_imgs)
        y2 = self.model(mid_imgs)
        y3 = self.model(large_imgs)

        return y1, y2, y3


def forward_features(model, x):
    _b = x.shape[0]

    # Googlenet features
    features = model

    # Get intermediate features after specific layers
    x1 = features.conv1(x)
    x1 = features.maxpool1(x1)
    x1 = features.conv2(x1)

    x2 = features.conv3(x1)
    x2 = features.maxpool2(x2)

    x3 = features.inception3a(x2)
    x3 = features.inception3b(x3)

    x3 = features.maxpool3(x3)

    x4 = features.inception4a(x3)

    x4 = features.inception4b(x4)
    x4 = features.inception4c(x4)
    x4 = features.inception4d(x4)
    x4 = features.inception4e(x4)
    x4 = features.maxpool4(x4)

    x5 = features.inception5b(x4)
    x5 = features.avgpool(x5)

    return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1), x5.view(_b, -1)


class MSNetPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = BaselineNet()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()


def create_random_subset(dataset, dataset_size):
    total_dataset_size = len(dataset)
    random_indices = torch.randperm(total_dataset_size)[:dataset_size]
    random_subset = torch.utils.data.Subset(dataset, random_indices)
    return random_subset


def main():
    DATA_ROOT = '/share/wenzhuoliu/torch_ds/imagenet-subset/val'
    val_ckpt_path = '/share/wenzhuoliu/code/test-code/CKA/supervised-ckpt/supervised-baseline.ckpt'
    # model = resnet50(pretrained=True)
    # model = MSNetPL.load_from_checkpoint(checkpoint_path=val_ckpt_path, args=None).encoder.model
    model = MSNetPL(args=None).encoder.model

    batch_size = 128
    dataset_size = 128
    num_sweep = 1
    num_features = 5
    small_size = 32
    large_size = 224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    perms = [torch.randperm(dataset_size) for _ in range(num_sweep)]
    dataset = datasets.ImageFolder(DATA_ROOT, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    dataset = create_random_subset(dataset, dataset_size)

    model.cuda()
    model.eval()
    cka_logger = CKA_Minibatch_Grid(num_features, num_features)
    with torch.no_grad():
        for sweep in range(num_sweep):
            dataset_sweep = torch.utils.data.Subset(dataset, perms[sweep])
            data_loader = torch.utils.data.DataLoader(
                dataset_sweep,
                batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True)
            for images, targets in tqdm(data_loader):
                images = images.cuda()

                images_small = F.interpolate(images, size=small_size, mode='bilinear')
                images_small = F.interpolate(images_small, size=large_size, mode='bilinear')
                features1 = forward_features(model, images_small)
                features2 = forward_features(model, images)
                cka_logger.update(features1, features2)
                # cka_logger.update(features1, features1)
                torch.cuda.empty_cache()

    cka_matrix = cka_logger.compute()
    cka_diag = np.diag(cka_matrix)
    return cka_diag


if __name__ == '__main__':
    results = []
    num_executions = 10
    # torch.random.manual_seed(0)

    for _ in range(num_executions):
        result = main()
        results.append(result)

    formatted_results = []
    for result in results:
        formatted_result = ', '.join([str(val) for val in result])
        formatted_results.append(f"[{formatted_result}]")

    print(', '.join(formatted_results))