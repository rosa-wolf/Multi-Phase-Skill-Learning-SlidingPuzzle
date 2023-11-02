import numpy as np
import torch
from itertools import permutations
import argparse
import time
import csv

from visualize_transitions import visualize_transition


# enumerate skills
#SKILLS = np.array([[0, 1], [0, 3], [2, 1], [2, 5], [3, 0], [3, 4], [5, 2],
#                   [5, 4], [1, 0], [1, 2], [1, 4], [4, 1], [4, 3], [4, 5]])

SKILLS = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                   [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])

# list of neighbors for each field from 0 to 5
neighbors = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]


if __name__ == '__main__':
    # go through all possible initial states where skill execution is possible and where it isn't possible
    # write to file in form (one-hot encoding, init_state, goal_state)
    with open('../transitions/transitions_change.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        for skill in range(14):
            print("skill = ", skill)
            fields = np.array([0, 1, 2, 3, 4, 5])
            # get one-hot encoding of skill
            one_hot = np.zeros((14,))
            one_hot[skill] = 1

            # we now get as input one-hot encodign of empty field and one-hot encoding of skill
            # and as output one-hot encoding of empty field after transitioning with skill

            # for each skill there is only one input, where change happens
            empty_field = SKILLS[skill, 1]
            new_empty_field = SKILLS[skill, 0]

            # for all other inputs skill has zero effect
            # sample transitions
            for _ in range(100):
                input = np.random.choice(fields)
                if input == empty_field:
                    output = new_empty_field
                else:
                    output = input

                file_input = np.zeros((6,))
                file_input[input] = 1
                file_output = np.zeros((6,))
                file_output[output] = 1
                # append transition to file
                writer.writerow(np.concatenate([file_input, one_hot, file_output]))
