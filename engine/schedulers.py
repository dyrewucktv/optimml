class IdentityLR(object):
    """Learning rate scheduler that keeps the learning rate constant."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        pass
