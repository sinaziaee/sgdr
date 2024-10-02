import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestartsLR(_LRScheduler):
    def __init__(self, optimizer, initial_lr: 1e-1, T_0: int = 50, T_mult: int = 2, eta_min: float = 1e-4, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.T_0 = T_0  # Initial number of iterations for the first restart
        self.T_mult = T_mult  # Factor by which the period grows after each restart
        self.eta_min = eta_min  # Minimum learning rate
        self.T_cur = 0 if current_step is None else current_step  # Current number of iterations since last restart
        
        # Number of iterations since the last restart
        self.current_period = self.T_0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.T_cur
            print("----- initial step")

        self.T_cur += 1

        # If current period is complete, reset and multiply by T_mult
        if current_step >= self.current_period:
            print("----- in multiplying")
            # print("current_step:", current_step, "- current_period:", self.current_period)
            self.T_cur = current_step - self.current_period
            self.current_period *= self.T_mult

        # Cosine annealing
        print("============= T_cur", self.T_cur)
        print("============= current_step:", current_step, "current_period", self.current_period)
        cosine_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                    (1 + math.cos(math.pi * self.T_cur / self.current_period)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cosine_lr
        print("cosine lr:", cosine_lr)

# Example usage
# Assuming you have an optimizer already created, e.g., optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
# initial_lr = 0.1
# T_0 = 10  # Number of epochs before the first restart
# T_mult = 2  # The number of epochs grows by a factor of 2 after each restart
# scheduler = CosineAnnealingWarmRestartsLR(optimizer, initial_lr, T_0, T_mult)
