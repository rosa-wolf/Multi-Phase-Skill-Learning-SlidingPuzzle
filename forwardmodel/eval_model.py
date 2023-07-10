import torch
import time
import numpy as np
import math
import os


from forward_model import ForwardModel

# load best model
my_forwardmodel = ForwardModel(batch_size=60, learning_rate=0.001, precision='float64')
my_forwardmodel.model = torch.load("models/best_model")
my_forwardmodel.model.eval()
'''
 skill 0 should push any piece from field 1 to zero if possible (or do nothing)
 but it pushes from field 1 to any free neighboring field

 => make sure that in training there are transitions where neighboring fields are free, but not the one we want to push to (done does not solve problem)
 => formulate skills differently? 
       - Then we have to have symbolic observation in agents observation because actions would be highly dependent on symbolic observation
       - we would have less skills

 the accuracy returns 1 iff the bernoulli state of the prediction is identical to the true value of the symbolic observation
'''

# test single transitions
skill = 13
one_hot = np.zeros((14,))
one_hot[skill] = 1

# push piece 3 from field 1 to field zero
sym_obs = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
goal_obs = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

y = torch.from_numpy(goal_obs.flatten())[None, :]

input = np.concatenate((sym_obs.flatten(), one_hot))
x = torch.from_numpy(input)
x = x[None, :]
# get models prediction
y_pred = my_forwardmodel.model(x)
old_shape = y_pred.shape
y_pred = y_pred.reshape((my_forwardmodel.pieces, my_forwardmodel.width * my_forwardmodel.height))
y_pred = torch.softmax(y_pred, dim=1)
y_pred = y_pred.reshape(old_shape)

print(y_pred.reshape((5, 6)))
print("==================================")

acc = my_forwardmodel.calculate_accuracy(y_pred, y)
print("acc = ", acc)

loss = my_forwardmodel.criterion(y_pred, y)

print("loss = ", loss)

print("===================================")

succ = torch.round(y_pred.reshape((5, 6)))

print(succ)


