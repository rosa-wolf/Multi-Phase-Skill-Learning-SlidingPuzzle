import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int):
        super().__init__()

        self.input = nn.Linear(input_size, 256)
        self.hidden = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_size)

    def forward(self, x):
        """
        x: input with shape(batch_size, sym_obs_size + num_skills)
        """
        h_1 = F.relu(self.input(x))

        # h_1 = [batch_size, 256]

        h_2 = F.relu(self.hidden(h_1))

        # h_2 = [batch_size, 256]

        y_pred = self.output(h_2)

        # y_pred = [batch_size, sym_obs_size]

        return y_pred
