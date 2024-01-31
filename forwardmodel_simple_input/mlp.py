import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,
                 width: int,
                 height: int,
                 num_skills: int):
        super().__init__()

        self.width = width
        self.height = height
        self.num_skills = num_skills


        output_size = self.width * self.height
        input_size = output_size + self.num_skills

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

        # interpret y_pred directly as one-hot encoding of block placements for all blocks
        # do multiclass classification
        # each row has to some up to one
        y_pred = torch.softmax(y_pred, axis=1)
        # get it back to previous shape

        if (torch.isnan(y_pred)).any():
            print("y_pred contains nan values")

        return y_pred
