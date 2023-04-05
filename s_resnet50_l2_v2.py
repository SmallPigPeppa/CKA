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


def unified_net():
    u_net = resnet50(pretrained=False)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    return u_net


class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), resnet50(pretrained=False).layer1
        )
        self.unified_net = unified_net()
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)
        self.unified_size = (56, 56)

    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)

        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')

        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)

        return z1, z2, z3, y1, y2, y3


def forward_features_small(model, x):
    _b = x.shape[0]

    # Initial layers
    x0 = model.small_net[0](x)
    x0 = model.small_net[1](x0)
    x0 = model.small_net[2](x0)

    # ResNet50 layers
    x1_0 = model.small_net[3][0](x0)
    x1_1 = model.small_net[3][1](x1_0)
    x1_2 = model.small_net[3][2](x1_1)
    x1_2 = F.interpolate(x1_2, size=model.unified_size, mode='bilinear')

    x2_0 = model.unified_net.layer2[0](x1_2)
    x2_1 = model.unified_net.layer2[1](x2_0)
    x2_2 = model.unified_net.layer2[2](x2_1)
    x2_3 = model.unified_net.layer2[3](x2_2)

    x3_0 = model.unified_net.layer3[0](x2_3)
    x3_1 = model.unified_net.layer3[1](x3_0)
    x3_2 = model.unified_net.layer3[2](x3_1)
    x3_3 = model.unified_net.layer3[3](x3_2)
    x3_4 = model.unified_net.layer3[4](x3_3)
    x3_5 = model.unified_net.layer3[5](x3_4)

    x4_0 = model.unified_net.layer4[0](x3_5)
    x4_1 = model.unified_net.layer4[1](x4_0)
    x4_2 = model.unified_net.layer4[2](x4_1)

    return [x0.view(_b, -1), x1_2.view(_b, -1), x2_3.view(_b, -1), x3_5.view(_b, -1), x4_2.view(_b, -1)]


def forward_features_large(model, x):
    _b = x.shape[0]

    # Initial layers
    x0 = model.large_net[0](x)
    x0 = model.large_net[1](x0)
    x0 = model.large_net[2](x0)
    x0 = model.large_net[3](x0)

    # ResNet50 layers
    x1_0 = model.large_net[4][0](x0)
    x1_1 = model.large_net[4][1](x1_0)
    x1_2 = model.large_net[4][2](x1_1)
    x1_2 = F.interpolate(x1_2, size=model.unified_size, mode='bilinear')

    x2_0 = model.unified_net.layer2[0](x1_2)
    x2_1 = model.unified_net.layer2[1](x2_0)
    x2_2 = model.unified_net.layer2[2](x2_1)
    x2_3 = model.unified_net.layer2[3](x2_2)

    x3_0 = model.unified_net.layer3[0](x2_3)
    x3_1 = model.unified_net.layer3[1](x3_0)
    x3_2 = model.unified_net.layer3[2](x3_1)
    x3_3 = model.unified_net.layer3[3](x3_2)
    x3_4 = model.unified_net.layer3[4](x3_3)
    x3_5 = model.unified_net.layer3[5](x3_4)

    x4_0 = model.unified_net.layer4[0](x3_5)
    x4_1 = model.unified_net.layer4[1](x4_0)
    x4_2 = model.unified_net.layer4[2](x4_1)

    return [x0.view(_b, -1), x1_2.view(_b, -1), x2_3.view(_b, -1), x3_5.view(_b, -1), x4_2.view(_b, -1)]


class MSNetPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = MultiScaleNet()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()


def create_random_subset(dataset, dataset_size):
    total_dataset_size = len(dataset)
    random_indices = torch.randperm(total_dataset_size)[:dataset_size]
    random_subset = torch.utils.data.Subset(dataset, random_indices)
    return random_subset


def main():
    DATA_ROOT = '/share/wenzhuoliu/torch_ds/imagenet-subset/val'
    val_ckpt_path = '/share/wenzhuoliu/code/test-code/CKA/supervised-ckpt/supervised-l2.ckpt'
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

    # model = resnet50(pretrained=True)
    model = MSNetPL.load_from_checkpoint(checkpoint_path=val_ckpt_path, args=None).encoder
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
                # images_small = F.interpolate(images_small, size=large_size, mode='bilinear')
                features1 = forward_features_small(model, images_small)
                features2 = forward_features_large(model, images)
                cka_logger.update(features1, features2)
                # cka_logger.update(features1, features1)
                torch.cuda.empty_cache()

    cka_matrix = cka_logger.compute()
    cka_diag = np.diag(cka_matrix)
    return cka_diag


if __name__ == '__main__':
    results = []
    num_executions = 10
    torch.random.manual_seed(0)

    for _ in range(num_executions):
        result = main()
        results.append(result)

    formatted_results = []
    for result in results:
        formatted_result = ', '.join([str(val) for val in result])
        formatted_results.append(f"[{formatted_result}]")

    print(', '.join(formatted_results))
