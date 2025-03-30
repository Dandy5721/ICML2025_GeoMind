from torch.optim.optimizer import Optimizer
import torch


class MyOptimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0

        loss = self.optimizer.step(closure)
        return loss
