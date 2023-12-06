import torch
import time
import numpy as np
import math
import os


from forwardmodel_simple_input.forward_model import ForwardModel

# load model

fm = ForwardModel(width=2,
                      height=1,
                      num_skills=2,
                      batch_size=10,
                      learning_rate=0.001)

# save model
fm_path = "/home/rosa/Documents/Uni/Masterarbeit/SEADS_SlidingPuzzle/models/fm_trained-with-policy_from_zero_03"
fm.model.load_state_dict(torch.load(fm_path))

input_states = [np.array([0, 1]), np.array([1, 0])]
input_skills = [0, 1]

for k in input_skills:
    print(f"====================\nskill = {k}")
    for state in input_states:
        one_hot_skill = np.zeros(2)
        one_hot_skill[k] = 1

        pred = fm.get_p_matrix(state, one_hot_skill)
        print(f"--------------\nstate = {state}\npred = {pred}")

        for succ in input_states:
            reward = fm.calculate_reward(state, succ, k)
            print(f"reward for going from state {state} to {succ}: {reward}")
        print("\n")