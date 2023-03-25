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



class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.resnet=resnet50(pretrained=False)
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)


    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        small_imgs = F.interpolate(small_imgs, size=self.large_size, mode='bilinear')
        mid_imgs = F.interpolate(mid_imgs, size=self.large_size, mode='bilinear')



        y1 = self.resnet(small_imgs)
        y2 = self.resnet(mid_imgs)
        y3 = self.resnet(large_imgs)


        return y1, y2, y3


def forward_features(model, x):
    _b = x.shape[0]

    # Initial layers
    x0 = model.conv1(x)
    x0 = model.bn1(x0)
    x0 = model.relu(x0)
    x0 = model.maxpool(x0)

    # ResNet50 layers
    x1_0 = model.layer1[0](x0)
    x1_1 = model.layer1[1](x1_0)
    x1_2 = model.layer1[2](x1_1)

    x2_0 = model.layer2[0](x1_2)
    x2_1 = model.layer2[1](x2_0)
    x2_2 = model.layer2[2](x2_1)
    x2_3 = model.layer2[3](x2_2)

    x3_0 = model.layer3[0](x2_3)
    x3_1 = model.layer3[1](x3_0)
    x3_2 = model.layer3[2](x3_1)
    x3_3 = model.layer3[3](x3_2)
    x3_4 = model.layer3[4](x3_3)
    x3_5 = model.layer3[5](x3_4)

    x4_0 = model.layer4[0](x3_5)
    x4_1 = model.layer4[1](x4_0)
    x4_2 = model.layer4[2](x4_1)

    return [x0.view(_b, -1), x1_0.view(_b, -1), x1_1.view(_b, -1), x1_2.view(_b, -1),
            x2_0.view(_b, -1), x2_1.view(_b, -1), x2_2.view(_b, -1), x2_3.view(_b, -1),
            x3_0.view(_b, -1), x3_1.view(_b, -1), x3_2.view(_b, -1), x3_3.view(_b, -1), x3_4.view(_b, -1), x3_5.view(_b, -1),
            x4_0.view(_b, -1), x4_1.view(_b, -1), x4_2.view(_b, -1)]



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
    batch_size = 128
    dataset_size = 1280
    num_sweep = 10
    num_features = 17
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
    model = MSNetPL.load_from_checkpoint(checkpoint_path=val_ckpt_path, args=None).encoder.resnet
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
    num_executions = 5
    # torch.random.manual_seed(0)

    for _ in range(num_executions):
        result = main()
        results.append(result)

    formatted_results = []
    for result in results:
        formatted_result = ', '.join([str(val) for val in result])
        formatted_results.append(f"[{formatted_result}]")

    print(', '.join(formatted_results))
