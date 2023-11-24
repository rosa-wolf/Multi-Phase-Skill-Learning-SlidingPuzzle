import torch
import time
import numpy as np
import math
import os


from forwardmodel_simple_input.forward_model import ForwardModel


SKILLS = np.array([[1, 0], [0, 1]])
def visualize_result(states, skills):
     states = np.array(states)
     states = states.reshape((states.shape[0], 1, 2))

     for i, state in enumerate(states):
         print("| {} | {} |".format(np.where(state[:, 1] == 1)[0],
                                    np.where(state[:, 0] == 1)[0]))
         if i < len(skills):
            print("------------------------------------------")
            print("----> skill: {}, intended effect: {}".format(skills[i], SKILLS[skills[i]]))
            print("------------------------------------------")

if __name__ == "__main__":
        # load forward model (untrained)
    fm = ForwardModel(width=2,
                      height=1,
                      num_skills=2,
                      batch_size=10,
                      learning_rate=0.001,
                      precision='float64')

    # load best model
    fm.model.load_state_dict(torch.load("models/fm_trained-with-policy"))
    fm.model.eval()

    # test whether path-planning works for very simple problem
    init_state = np.array([[1, 0]])
    goal_state = np.array([[0, 1]])

    states, skills = fm.breadth_first_search(init_state.flatten(), goal_state.flatten())
    print("skills = ", skills)
    print("states = ", states)

    visualize_result(states, skills)