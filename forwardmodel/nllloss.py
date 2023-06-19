import torch
import torch.nn as nn


class NLLLoss_customized(nn.Module):
    def __init__(self):
        super(NLLLoss_customized, self).__init__()

    def forward(self, x, y):
        """
        @param x: model output (shape: batch_size x sym_obs_size)( predicted probability vector)
        @param y: target/true value (real probability vector)

        @return l: cross entropy loss
        """
        # problem if everything is correct model input and output have only 2 different numbers
        # loss very small even if model output not different from model input
        #l = - torch.nansum(y * torch.log(x) + (1 - y) * torch.log(1 - x))


        l = - torch.nansum(y * torch.log(x))

        # average over batches
        num_batches = x.shape[0]
        l /= num_batches

        return l
