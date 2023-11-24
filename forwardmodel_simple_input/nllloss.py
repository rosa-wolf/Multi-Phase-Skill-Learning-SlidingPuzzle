import torch
import torch.nn as nn
from .visualize_transitions import visualize_transition


class NLLLoss_customized(nn.Module):
    def __init__(self):
        super(NLLLoss_customized, self).__init__()

    def forward(self, x, y):
        """
        @param x: model output (shape: batch_size x #fields)( predicted probability vector)
        @param y: target/true value (real probability vector)

        @return l: cross entropy loss
        """
        # problem if everything is correct model input and output have only 2 different numbers
        # loss very small even if model output not different from model input
        #l = - torch.nansum(y * torch.log(x) + (1 - y) * torch.log(1 - x))

        # TODO: look which data point has largest loss

        # one-line
        l = - torch.nansum(y * torch.log(x))

        # average over batches
        num_batches = x.shape[0]
        l /= num_batches

        return l # sum_loss, max_loss, max_ep
