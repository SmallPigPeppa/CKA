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
    x0 = model.bn1(x0)
    x0 = model.relu(x0)
    x0 = model.maxpool(x0)

    # ResNet18 layers
    x1_0 = model.layer1[0](x0)
    x1_1 = model.layer1[1](x1_0)

    x2_0 = model.layer2[0](x1_1)
    x2_1 = model.layer2[1](x2_0)

    x3_0 = model.layer3[0](x2_1)
    x3_1 = model.layer3[1](x3_0)

    x4_0 = model.layer4[0](x3_1)
    x4_1 = model.layer4[1](x4_0)

    return [x0.view(_b, -1), x1_0.view(_b, -1), x1_1.view(_b, -1),
            x2_0.view(_b, -1), x2_1.view(_b, -1),
            x3_0.view(_b, -1), x3_1.view(_b, -1),
            x4_0.view(_b, -1), x4_1.view(_b, -1)]


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
    finetune_ckpt = '/share/wenzhuoliu/code/test-code/CKA-ISSL/experiments/2023_04_08_05_21_19-finetune/3t5542bb/finetune-task1-ep=499-3t5542bb.ckpt'
    batch_size = 128
    dataset_size = 128
    num_sweep = 1
    num_features = 9
    num_classes = 100
    num_tasks = 5

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    perms = [torch.randperm(dataset_size) for _ in range(num_sweep)]
    tasks = torch.randperm(num_classes).chunk(num_tasks)
    dataset = datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    dataset_old = split_dataset(dataset=dataset, task_idx=0, tasks=tasks)
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
