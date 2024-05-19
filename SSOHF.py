from torch.optim import Optimizer

class SSOHF(Optimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
