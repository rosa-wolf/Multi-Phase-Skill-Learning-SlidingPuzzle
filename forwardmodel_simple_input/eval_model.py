import torch
import time
import numpy as np
import math
import os


from forward_model import ForwardModel

# load best model
my_forwardmodel = ForwardModel(batch_size=60, learning_rate=0.001, precision='float64')
#my_forwardmodel.model = torch.load("models/best_model")
my_forwardmodel.model.load_state_dict(torch.load("models/best_model_change"))
my_forwardmodel.model.eval()

SKILLS = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                   [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])
def visualize_result(states, skills):
    states = np.array(states)
    states = states.reshape((states.shape[0], 5, 6))

    for i, state in enumerate(states):
        print("| {} | {} | {} |\n| {} | {} | {} |".format(np.where(state[:, 2] == 1)[0],
                                                          np.where(state[:, 1] == 1)[0],
                                                          np.where(state[:, 0] == 1)[0],
                                                          np.where(state[:, 5] == 1)[0],
                                                          np.where(state[:, 4] == 1)[0],
                                                          np.where(state[:, 3] == 1)[0]))
        if i < len(skills):
            print("------------------------------------------")
            print("----> skill: {}, intended effect: {}".format(skills[i], SKILLS[skills[i]]))
            print("------------------------------------------")

'''
 skill 0 should push any piece from field 1 to zero if possible (or do nothing)
 but it pushes from field 1 to any free neighboring field

 => make sure that in training there are transitions where neighboring fields are free, but not the one we want to push to (done does not solve problem)
 => formulate skills differently? 
       - Then we have to have symbolic observation in agents observation because actions would be highly dependent on symbolic observation
       - we would have less skills

 the accuracy returns 1 iff the bernoulli state of the prediction is identical to the true value of the symbolic observation
'''
"""
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
"""
# test whether path-planning works for very simple problem
#init_state = np.array([[1, 0, 0, 0, 0, 0],
#                       [0, 1, 0, 0, 0, 0],
#                       [0, 0, 1, 0, 0, 0],
#                       [0, 0, 0, 1, 0, 0],
#                       [0, 0, 0, 0, 1, 0]])
#goal_state = np.array([[1, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 1, 0, 0],
#                       [0, 0, 1, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 1],
#                       [0, 1, 0, 0, 0, 0]])


init_state = np.array([[0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1]])
goal_state = np.array([[0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 0]])
states, skills = my_forwardmodel.breadth_first_search(init_state.flatten(), goal_state.flatten())
print("number of moves = ", len(states) - 1)
visualize_result(states, skills)
