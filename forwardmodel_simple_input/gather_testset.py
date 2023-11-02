import numpy as np
import torch
from itertools import permutations
import argparse
import time
import csv


# enumerate skills
#SKILLS = np.array([[0, 1], [0, 3], [2, 1], [2, 5], [3, 0], [3, 4], [5, 2],
#                   [5, 4], [1, 0], [1, 2], [1, 4], [4, 1], [4, 3], [4, 5]])

SKILLS = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                   [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])

# list of neighbors for each field from 0 to 5
neighbors = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]


if __name__ == '__main__':

    # write to file in former (one-hot encoding, init_state, goal_state)
    with open('../transitions/test_transitions.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        for skill in range(14):
            fields = np.array([0, 1, 2, 3, 4, 5])
            # get one-hot encoding of skill
            one_hot = np.zeros((14,))
            one_hot[skill] = 1

            # only add transitions where neighboring field we do not want to push to is empty
            for free in neighbors[SKILLS[skill, 0]]:
                if free != SKILLS[skill, 1]:
                    fields_poss = np.delete(fields, free)
                    perms = permutations(fields_poss)
                    for order in perms:
                        # set board in env to initial state
                        init_state = np.zeros((5, 6))
                        for i in range(5):
                            init_state[i, order[i]] = 1
                        # get goal state
                        goal_state = init_state.copy()

                        writer.writerow(np.concatenate([init_state.flatten(), one_hot, goal_state.flatten()]))

            ## put all possible transitions into training data where initially empty field is the one we want to push from
            #fields_imposs = np.delete(fields, SKILLS[skill, 0])
            #perms = permutations(fields_imposs)
            #for order in perms:
            #    # set board in env to initial state
            #    init_state = np.zeros((5, 6))
            #    for i in range(5):
            #        init_state[i, order[i]] = 1
            #    # goal state similar to initial state
            #    # write to file
            #    writer.writerow(np.concatenate([init_state.flatten(), one_hot, init_state.flatten()]))





