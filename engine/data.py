from typing import Literal

import torch
from torchvision import datasets, transforms

TRANSFORMS = {
    "lenet": transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    "mlp": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x, start_dim=0)),
        ]
    ),
}


def get_loaders(network: Literal["lenet", "mlp"]):
    transform = TRANSFORMS[network]
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
    )
    return train_loader, test_loader
