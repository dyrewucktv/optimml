from torch.optim.lr_scheduler import LRScheduler
import torch


class IdentityLR(object):
    """Learning rate scheduler that keeps the learning rate constant."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        pass


class SuperLRScheduler(LRScheduler):
    def _initial_step(self):
        self.cur_weights = None  # current weights
        self.weights_1 = None  # weights in last epoch
        self.weights_2 = None  # weights in epoch before last epoch
        super()._initial_step()

    def get_lr(self):
        lrs = self.base_lrs
        if self.weights_1 and self.weights_2:
            lrs = []
            # w1 weights in last epoch
            # w2 weights in epoch before last
            # c current weights
            for w1s, w2s, cs, lr in zip(
                self.weights_1,
                self.weights_2,
                self.cur_weights,
                self.base_lrs
            ):
                noms = []
                denoms = []
                for w1, w2, c in zip(w1s, w2s, cs):
                    noms.append(torch.sum(torch.abs(w1 - c)).item())
                    denoms.append(
                        max(
                                torch.sum(torch.abs(2 * w1 - c - w2)).item(),
                                1e-5
                            )
                        )
                lrs.append(lr * sum(noms) / sum(denoms))

        # print(self.base_lrs, lrs)
        return lrs

    def step(self, epoch=None):
        super().step(epoch)
        self.weights_2 = self.weights_1
        self.weights_1 = self.cur_weights
        self.cur_weights = self._get_cur_weights()

    def _get_cur_weights(self):
        return [
                [
                    param.detach().clone()
                    for param in group["params"]
                ]
                for group in self.optimizer.param_groups
            ]
