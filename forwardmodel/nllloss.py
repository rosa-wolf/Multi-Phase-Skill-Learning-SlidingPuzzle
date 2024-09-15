import torch
import torch.nn as nn
from .visualize_transitions import visualize_transition


class NLLLoss_customized(nn.Module):
    def __init__(self):
        super(NLLLoss_customized, self).__init__()

    def forward(self, x, y):
        """
        @param x: model output (shape: batch_size x sym_obs_size)( predicted probability vector)
        @param y: target/true value (real probability vector)

        @return l: cross entropy loss
        """

        # one-line
        l = - torch.nansum(y * torch.log(x))

        # average over batches
        num_batches = x.shape[0]
        l /= num_batches

        return l
