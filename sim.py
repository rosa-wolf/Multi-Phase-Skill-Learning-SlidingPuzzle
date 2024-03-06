import numpy as np
from generate_goal import generate_goal


if __name__ == '__main__':

    init_sym_state = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]])

    init_sym_state = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0]])

    generate_goal(init_sym_state, depth=4, puzzle_size=[2, 3])
