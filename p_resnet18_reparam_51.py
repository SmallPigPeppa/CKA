from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset, Subset
from cka import CKA_Minibatch_Grid
import numpy as np
from tqdm import tqdm
from models.resnet18_cifar_reparam import resnet18


def load_ckpt(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder.", "")] = state[k]
        del state[k]
    return state


def forward_features(model, x):
    _b = x.shape[0]

    # Initial layers
    x0 = model.conv1(x)
    x1 = model.bn1(x0)
    x2 = model.relu(x1)
    x3 = model.maxpool(x2)

    features = [x0, x1, x3]

    # ResNet18 layers
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]

    for layer in layers:
        for block in layer:
            x3 = block.conv1(x3)
            x4 = block.bn1(x3)
            x5 = block.relu(x4)
            x6 = block.conv2(x5)
            x7 = block.bn2(x6)

            x8 = x3 + x7  # Use + for element-wise addition
            x3 = block.relu(x8)

            features += [x3, x4, x5, x6, x7, x8]

    # Flatten and concatenate all the feature maps
    flattened_features = [feat.view(_b, -1) for feat in features]

    return flattened_features


def create_random_subset(dataset, dataset_size):
    total_dataset_size = len(dataset)
    random_indices = torch.randperm(total_dataset_size)[:dataset_size]
    random_subset = torch.utils.data.Subset(dataset, random_indices)
    return random_subset


def split_dataset(dataset, task_idx, tasks):
    mask = [(c in tasks[task_idx]) for c in dataset.targets]
    indexes = torch.tensor(mask).nonzero()
    task_dataset = Subset(dataset, indexes)
    return task_dataset


def main(dataset):
    dataset = create_random_subset(dataset, dataset_size)
    joint_model = resnet18()
    joint_model.fc = nn.Identity()
    joint_state = load_ckpt(joint_ckpt)
    joint_model.load_state_dict(joint_state, strict=True)
    joint_model.cuda()
    joint_model.eval()

    finetune_model = resnet18()
    finetune_model.fc = nn.Identity()
    finetune_state = load_ckpt(finetune_ckpt)
    finetune_model.load_state_dict(finetune_state, strict=True)
    finetune_model.cuda()
    finetune_model.eval()

    cka_logger = CKA_Minibatch_Grid(num_features, num_features)
    with torch.no_grad():
        for sweep in range(num_sweep):
            dataset_sweep = Subset(dataset, perms[sweep])
            data_loader = torch.utils.data.DataLoader(
                dataset_sweep,
                batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True)
            for images, targets in tqdm(data_loader):
                images = images.cuda()
                features1 = forward_features(joint_model, images)
                # import pdb;pdb.set_trace()
                features2 = forward_features(finetune_model, images)
                cka_logger.update(features1, features2)
                torch.cuda.empty_cache()

    cka_matrix = cka_logger.compute()
    cka_diag = np.diag(cka_matrix)
    return cka_diag


if __name__ == '__main__':
    seed_everything(5)
    DATA_ROOT = '/share/wenzhuoliu/torch_ds'
    joint_ckpt = '/share/wenzhuoliu/code/test-code/CKA-ISSL/experiments/2023_04_08_07_20_09-upbound/3i4ve44h/upbound-task1-ep=499-3i4ve44h.ckpt'
    # joint_ckpt = '/share/wenzhuoliu/code/test-code/CKA-ISSL/experiments/2023_04_08_05_21_19-finetune/3qlk4687/finetune-task0-ep=499-3qlk4687.ckpt'
    # finetune_ckpt = '/share/wenzhuoliu/code/test-code/CKA-ISSL/experiments/2023_04_08_05_21_19-finetune/3t5542bb/finetune-task1-ep=499-3t5542bb.ckpt'
    # finetune_ckpt = '/share/wenzhuoliu/code/test-code/CKA-ISSL/experiments/2023_04_08_05_21_19-finetune/3qlk4687/finetune-task0-ep=499-3qlk4687.ckpt'
    finetune_ckpt = 'experiments/2023_04_08_07_22_42-reparam/pxbdry3u/reparam-task1-ep=499-pxbdry3u.ckpt'
    batch_size = 128
    dataset_size = 128
    num_sweep = 1
    num_features = 51
    num_classes = 100
    num_tasks = 5

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    perms = [torch.randperm(dataset_size) for _ in range(num_sweep)]
    tasks = [
        [11, 22, 39, 23, 42, 30, 78, 81, 64, 20, 29, 79, 15, 69, 86, 63, 55, 53,
         73, 68],
        [89, 67, 58, 97, 96, 92, 37, 14, 75, 51, 54, 7, 3, 6, 50, 40, 45, 4,
         83, 98],
        [27, 12, 8, 99, 60, 87, 28, 5, 84, 34, 82, 16, 72, 49, 59, 31, 71, 35,
         66, 76],
        [61, 17, 36, 62, 13, 2, 38, 94, 80, 19, 25, 18, 0, 1, 46, 74, 85, 91,
         52, 77],
        [21, 33, 32, 88, 93, 70, 44, 47, 26, 57, 90, 95, 48, 65, 43, 10, 9, 56,
         24, 41]
    ]
    dataset = datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    dataset_old = split_dataset(dataset=dataset, task_idx=1, tasks=tasks)
    results = []
    num_executions = 10
    for _ in range(num_executions):
        result = main(dataset_old)
        results.append(result)

    formatted_results = []
    for result in results:
        formatted_result = ', '.join([str(val) for val in result])
        formatted_results.append(f"[{formatted_result}]")

    print(', '.join(formatted_results))
