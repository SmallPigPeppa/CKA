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




def forward_features(model, x):
    _b = x.shape[0]

    # Initial layers
    x = model.conv1(x)
    features = [x.view(_b, -1)]
    x = model.bn1(x)
    # features.append(x.view(_b, -1))
    x = model.relu(x)
    x = model.maxpool(x)
    # features.append(x.view(_b, -1))

    # Iterate through ResNet layers and blocks
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            identity = x
            x = block.conv1(x)
            features.append(x.view(_b, -1))
            x = block.bn1(x)
            x = block.relu(x)
            x = block.conv2(x)
            features.append(x.view(_b, -1))
            x = block.bn2(x)
            x = block.relu(x)
            x = block.conv3(x)
            # features.append(x.view(_b, -1))
            x = block.bn3(x)
            if block.downsample is not None:
                identity = block.downsample(identity)

            x += identity
            x = block.relu(x)
            features.append(x.view(_b, -1))



    return features





#
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
    num_features = 49
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
    plt.savefig('resnet50-3.pdf')
    print(cka_diag)

if __name__ == '__main__':
    main()