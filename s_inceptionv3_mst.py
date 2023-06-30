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
from torchvision.models import vgg16,inception_v3

from pytorch_lightning import LightningModule
PRETRAINED = True

class DenseNet121_L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.unified_net = inception_v3(pretrained=PRETRAINED)


def forward_features(model, x):
    _b = x.shape[0]

    # InceptionV3 features
    features = model

    # Get intermediate features after specific layers
    x1 = features.Conv2d_1a_3x3(x)
    x1 = features.Conv2d_2a_3x3(x1)
    x1 = features.Conv2d_2b_3x3(x1)
    x1 = features.maxpool1(x1)

    x2 = features.Conv2d_3b_1x1(x1)
    x2 = features.Conv2d_4a_3x3(x2)
    x2 = features.maxpool2(x2)

    x3 = features.Mixed_5b(x2)
    x3 = features.Mixed_5c(x3)
    x3 = features.Mixed_5d(x3)

    x4 = features.Mixed_6a(x3)
    x4 = features.Mixed_6b(x4)
    x4 = features.Mixed_6c(x4)
    x4 = features.Mixed_6d(x4)
    x4 = features.Mixed_6e(x4)

    x5 = features.Mixed_7a(x4)

    return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1), x5.view(_b, -1)



class MSC(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = DenseNet121_L2()

def create_random_subset(dataset, dataset_size):
    total_dataset_size = len(dataset)
    random_indices = torch.randperm(total_dataset_size)[:dataset_size]
    random_subset = torch.utils.data.Subset(dataset, random_indices)
    return random_subset


def main():
    DATA_ROOT = '/share/wenzhuoliu/torch_ds/imagenet-subset/val'
    val_ckpt_path = '/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/mstrain/checkpoints/inceptionv3-mstrain/last.ckpt'
    # model = resnet50(pretrained=True)
    model = MSC.load_from_checkpoint(checkpoint_path=val_ckpt_path, args=None).model.unified_net

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
