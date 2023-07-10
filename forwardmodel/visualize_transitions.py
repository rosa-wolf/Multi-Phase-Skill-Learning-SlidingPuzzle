import numpy as np
import torch
import csv

SKILLS = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                   [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])


def visualize_transition(start, skill, end):
    start = start.reshape((5, 6))
    end = end.reshape((5, 6))

    print("| {} | {} | {} |\n| {} | {} | {} |".format(np.where(start[:, 2] == 1)[0],
                                                      np.where(start[:, 1] == 1)[0],
                                                      np.where(start[:, 0] == 1)[0],
                                                      np.where(start[:, 5] == 1)[0],
                                                      np.where(start[:, 4] == 1)[0],
                                                      np.where(start[:, 3] == 1)[0]))

    skill = np.where(skill == 1)[0][0]
    print("------------------------------------------")
    print("----> skill: {}, intended effect: {}".format(skill, SKILLS[skill]))
    print("------------------------------------------")

    print("| {} | {} | {} |\n| {} | {} | {} |".format(np.where(end[:, 2] == 1)[0],
                                                      np.where(end[:, 1] == 1)[0],
                                                      np.where(end[:, 0] == 1)[0],
                                                      np.where(end[:, 5] == 1)[0],
                                                      np.where(end[:, 4] == 1)[0],
                                                      np.where(end[:, 3] == 1)[0]))

    print("=================================================\n=================================================")


if __name__ == "__main__":
    # read out transitions from file and visualize them
    file = "../transitions/transitions.csv"
    data = torch.from_numpy(np.genfromtxt(file, delimiter=","))

    for i, trans in enumerate(data):
        print(i)
        if i >= 1381:
            visualize_transition(trans[:30], trans[30:44], trans[44:])
        if i > 1481:
            break

