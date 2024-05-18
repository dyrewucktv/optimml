from collections import OrderedDict

import torch
from torch import concat, nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detach_and_copy_to_cpu(state: dict) -> OrderedDict:
    res = OrderedDict()
    for k, v in state.items():
        res[k] = v.detach().cpu().clone()
    return res


def optimal_lr(
    i: OrderedDict, ii: OrderedDict, iii: OrderedDict, epsilon: float
) -> OrderedDict:
    # i - theta_i, ii - theta_i+1, iii - theta_i+2
    res = OrderedDict()
    for (
        (ki, vi),
        (_, vii),
        (_, viii),
    ) in zip(i.items(), ii.items(), iii.items()):
        nominator = vii - vi
        denominator = 2 * vii - vi - viii
        res[ki] = (
            epsilon
            * nominator.abs().sum()
            / max(denominator.abs().sum(), 1e-5)
        )
    return res


def get_accuracy(loader: DataLoader, model: nn.Module) -> float:
    predictions = []
    labels = []
    for X, y in loader:
        with torch.no_grad():
            predictions.append(model(X.to(device)).argmax(dim=1))
            labels.append(y)
    return (
        (concat(predictions).to(device) == concat(labels).to(device))
        .float()
        .mean()
    )
