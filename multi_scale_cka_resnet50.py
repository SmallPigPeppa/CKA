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


# def forward_features(model, x):
#     _b = x.shape[0]
#     x = model.conv1(x)
#     x = model.bn1(x)
#     x = model.relu(x)
#     x = model.maxpool(x)
#
#     x = model.layer1(x)
#     x1 = x
#     x = model.layer2(x)
#     x2 = x
#     x = model.layer3(x)
#     x3 = x
#     x = model.layer4(x)
#     x4 = x
#     return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1)

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






# def forward_features(model, x):
#     _b = x.shape[0]
#     features = []
#
#     x = model.conv1(x)
#     x = model.bn1(x)
#     x = model.relu(x)
#     features.append(x.view(_b, -1))
#
#     x = model.maxpool(x)
#
#     for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
#         for block in layer:
#             x = block(x)
#             features.append(x.view(_b, -1))
#
#     return features

def main():
    DATA_ROOT = 'C:/Users/90532/Desktop/Datasets/imagent100/val'
    batch_size = 128
    dataset_size = 1280
    num_sweep = 10
    num_features = 17
    # num_features = 4
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

    model = resnet50(pretrained=True)
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
    plt.savefig('resnet50-2.pdf')

if __name__ == '__main__':
    main()