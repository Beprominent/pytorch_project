import torch
import torchvision.datasets as dsets
import numpy as np
import torch.utils.data.dataloader
from torchvision.transforms import transforms

def get_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=False, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    mean, std = get_mean_std(cifar10)
    print(mean, std)