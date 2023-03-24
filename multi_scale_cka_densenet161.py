from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.models import densenet161
import torchvision.transforms as transforms
import torch.nn.functional as F

from cka import CKA_Minibatch_Grid
import numpy as np

import torch
import torchvision.models as models


def forward_features(model, x):
    _b = x.shape[0]

    # DenseNet161 features
    features = model.features

    # Process input through initial convolution, batch norm and relu layers
    x = features.conv0(x)
    x = features.norm0(x)
    x = features.relu0(x)
    x = features.pool0(x)

    # Get intermediate features after each dense block
    x1 = features.denseblock1(x)
    x = features.transition1(x1)
    x2 = features.denseblock2(x)
    x = features.transition2(x2)
    x3 = features.denseblock3(x)
    x = features.transition3(x3)
    x4 = features.denseblock4(x)

    return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1)


def main():
    DATA_ROOT = 'C:/Users/90532/Desktop/Datasets/imagent100/val'
    batch_size = 128
    dataset_size = 1280
    num_sweep = 10
    num_features = 9
    num_features = 4
    small_size=32
    large_size=224



    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    torch.random.manual_seed(0)
    perms = [torch.randperm(dataset_size) for _ in range(num_sweep)]
    dataset = datasets.ImageFolder(DATA_ROOT, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    model = densenet161(pretrained=True)
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
                features1 = forward_features(model, images)
                features2 = forward_features(model, images_small)
                cka_logger.update(features1, features2)
                # cka_logger.update(features1, features1)
                torch.cuda.empty_cache()

    cka_matrix = cka_logger.compute()

    plt.title('Pretrained Resnet18 Layer CKA')
    # plt.xticks([0, 1, 2, 3], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])
    # plt.yticks([0, 1, 2, 3], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])
    # layers = [f"Layer {i + 1}" for i in range(num_features)]
    # plt.xticks(range(num_features), layers)
    # plt.yticks(range(num_features), layers)
    # plt.imshow(cka_matrix.numpy(), origin='lower', cmap='magma')
    cka_diag = np.diag(cka_matrix)

    # 绘制对角线
    plt.plot(cka_diag)
    # plt.clim(0, 1)
    # plt.colorbar()
    plt.savefig('densenet161.pdf')

if __name__ == '__main__':
    main()