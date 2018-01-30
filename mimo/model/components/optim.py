import numpy as np
import math


class ScheduledOptim(object):
    """ Learning rate scheduling """

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

        self.start_decay_at = 5
        self.initial_lr = 1e-4
        self.lr = self.initial_lr

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_current_lr(self):
        return self.lr

    def update_learning_rate(self):
        self.n_current_steps += 1

        epoch = self.n_current_steps/1000.
        if epoch > self.start_decay_at:
            self.lr = self.initial_lr * math.pow(0.99, epoch - self.start_decay_at)  # / math.sqrt(self.n_current_steps - self.start_decay_at)

        # todo: try simple decay, dropping warmup term
        #ndim = 512  #self.d_model
        #self.lr = np.power(ndim*16, -0.5) *  np.min([
        #    np.power(self.n_current_steps, -0.5),
        #    np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        #])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
