import torch
from torchvision import datasets, transforms
import random


def load_dataset(input_dataset, img_size, batch_size, percentage):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    if input_dataset == "MNIST":
        train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(
            './data', train=False, download=True, transform=transform)
    elif input_dataset == "FASHION_MNIST":
        train_dataset = datasets.FashionMNIST(
            './data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(
            './data', train=False, download=True, transform=transform)
    elif input_dataset == "CIFAR":
        train_dataset = datasets.CIFAR10(
            './data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            './data', train=False, download=True, transform=transform)
    indices = [i for i in list(range(len(train_dataset)))]
    random.shuffle(indices)
    select_num = int(len(indices) * percentage // 100)
    select_indices = indices[:select_num]
    train_dataset_sampler = torch.utils.data.SubsetRandomSampler(select_indices)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_dataset_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset.data), shuffle=True)
    print("Dataset size:", select_num)
    del train_dataset, test_dataset
    return train_loader, test_loader
