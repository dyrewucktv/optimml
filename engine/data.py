from typing import Literal

import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms

from .utils import device

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
        shuffle=False,
        num_workers=16,
    )
    test_Xs = torch.concat([batch[0] for batch in train_loader]).to(device)
    test_ys = torch.concat([batch[1] for batch in train_loader]).to(device)
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(test_Xs, test_ys),
        batch_size=128,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
    )
    test_Xs = torch.concat([batch[0] for batch in test_loader]).to(device)
    test_ys = torch.concat([batch[1] for batch in test_loader]).to(device)
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(test_Xs, test_ys),
        batch_size=64,
        shuffle=False,
    )
    return train_loader, test_loader
