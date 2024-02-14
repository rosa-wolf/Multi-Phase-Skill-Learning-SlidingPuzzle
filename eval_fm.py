import torch
import numpy as np


from forwardmodel_simple_input.forward_model import ForwardModel

# load model
num_skills = 8
fm = ForwardModel(num_skills=num_skills, batch_size=10, puzzle_size=[2, 2], learning_rate=0.001)

# save model
#fm_path = "/home/rosa/Documents/Uni/Masterarbeit/SEADS_SlidingPuzzle/SAC/checkpoints/2x3_puzzle/Parallel_Training/parallel2x3-new-penalty_num_skills3_sparseFalse_relabelingTrue/fm/fm"
fm_path = "/home/rosa/Documents/Uni/Masterarbeit/checkpoints/parallel2x2-new-penalty_num_skills8_sparseFalse_relabelingFalse/fm/fm"
fm.model.load_state_dict(torch.load(fm_path))

print(fm.get_full_pred())

#input_states = [np.array([0, 1]), np.array([1, 0])]
#input_skills = [0, 1]


input_states = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]

#input_states = [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
#                np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
#                np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
#                np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
#                np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
#                np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
#                np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
#                np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
#                np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])]

#input_states = [np.array([1, 0, 0, 0, 0, 0]),
#                np.array([0, 1, 0, 0, 0, 0]),
#                np.array([0, 0, 1, 0, 0, 0]),
#                np.array([0, 0, 0, 1, 0, 0]),
#                np.array([0, 0, 0, 0, 1, 0]),
#                np.array([0, 0, 0, 0, 0, 1]),]
for k in range(num_skills):
    print(f"====================\nskill = {k}")
    for state in input_states:
        one_hot_skill = np.zeros(num_skills)
        one_hot_skill[k] = 1

        pred = fm.get_p_matrix(state, one_hot_skill)
        print(f"--------------\nstate = {state}\npred = {pred}")

        for succ in input_states:
            reward = fm.calculate_reward(state, succ, k)
            print(f"reward for going from state {state} to {succ}: {reward}")
        print("\n")