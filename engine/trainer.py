from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .schedulers import (
    SuperLRScheduler,
    get_param_groups_with_names,
    make_super_scheduler_with_optimizer,
)
from .utils import detach_and_copy_to_cpu, device, optimal_lr


@dataclass(frozen=True)
class TrainingResults:
    losses: list[float]
    scheduler_lrs: list[float]
    optimal_lrs: list[dict[str, Tensor]]


class Trainer:

    def __init__(
        self,
        scheduler_factory: Callable[[Optimizer], LRScheduler],
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ):
        self.criterion = criterion
        self.scheduler_factory = scheduler_factory

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        num_epochs: int,
    ) -> TrainingResults:
        if self.scheduler_factory == make_super_scheduler_with_optimizer:
            optimizer, scheduler = make_super_scheduler_with_optimizer(model)
        else:
            optimizer = torch.optim.SGD(
                get_param_groups_with_names(model), lr=1e-2
            )
            scheduler = self.scheduler_factory(optimizer)

        losses = []
        weights = []
        scheduler_lrs = []
        optimal_lrs = []
        for epoch in (pbar := tqdm(range(num_epochs))):
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(images.to(device))
                loss = self.criterion(outputs, labels.to(device))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                if (i + 1) % 100 == 0:
                    self.log_to_pbar(pbar, epoch + 1, i + 1, loss)
                losses.append(loss.item())

                # Initial steps for super convergence
                if (
                    epoch == 0
                    and i in (10, 20)
                    and isinstance(scheduler, SuperLRScheduler)
                ):
                    scheduler.step()

                # Calculate optimal LR
                weights.append(detach_and_copy_to_cpu(model.state_dict()))
                epsilon = optimizer.param_groups[0]["lr"]
                scheduler_lrs.append(
                    {
                        group["name"]: group["lr"]
                        for group in optimizer.param_groups
                    }
                )
                if len(weights) > 3:
                    weights.pop(0)
                    optimal_lrs.append(
                        optimal_lr(
                            weights[-1], weights[-2], weights[-3], epsilon
                        )
                    )
            scheduler.step()

        return TrainingResults(
            losses=losses, scheduler_lrs=scheduler_lrs, optimal_lrs=optimal_lrs
        )

    def log_to_pbar(
        self, pbar: tqdm, epoch: int, step: int, loss: Tensor
    ) -> None:
        pbar.set_description(
            "Epoch [{}], Step [{}], Loss: {:.4f}".format(
                epoch, step, loss.item()
            )
        )
