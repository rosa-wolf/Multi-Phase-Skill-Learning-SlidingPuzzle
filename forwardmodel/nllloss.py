import torch
import torch.nn as nn


class NLLLoss_customized(nn.Module):
    def __init__(self):
        super(NLLLoss_customized, self).__init__()

    def forward(self, x, y):
        """
        @param x: model output (shape: batch_size x sym_obs_size)
        @param y: target/true value

        @return l: loss
        """

        l = - torch.sum(y * torch.log(x) + (1 - y) * torch.log(1 - x))

        # average over batches
        num_batches = x.shape[0]
        l /= num_batches

        return l
